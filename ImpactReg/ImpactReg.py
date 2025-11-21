from qt import QWidget
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import slicer
import json
import SimpleITK as sitk

from qt import (
    QWidget,QProcess, QIcon, QSize, QUrl, QDesktopServices, QPushButton
)
import os
import platform
import sitkUtils

from KonfAI import KonfAICoreWidget, AppTemplateWidget, Process, KonfAIMetricsPanel
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
import importlib
import re
import shutil

import numpy as np
from konfai.evaluator import Statistics

class ElastixProcess(Process):

    def __init__(self, _update_logs, _update_progress):
        super().__init__(_update_logs, _update_progress)
        self._total_iterations = 0

    def on_stdout_ready(self):
        line = self.readAllStandardOutput().data().decode().strip()
        if line:
            line = line.replace('\r\n', '\n').split('\r')[-1]
            self._update_logs(line)
            is_it = False
            for l in line.split("\n"):
                if re.match(r"^\d+", l):
                    parts = re.split(r"\s+", l)
                    try:
                        time_ms = float(parts[-1])
                        is_it = True
                        self._it += 1
                    except ValueError:
                        continue
            if is_it:
                self._update_progress(int(self._it/self._total_iterations*100), f"{time_ms:.2f} ms")
                        
    
    def set_total_iterations(self, total_iterations: int) -> None:
        self._total_iterations = total_iterations
        self._it = 0

#
# ImpactReg
#
class ImpactReg(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Impact Reg")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Registration")]
        self.parent.dependencies = ["KonfAI"]
        self.parent.contributors = [
            "Valentin Boussot (University of Rennes, France)",
            "Cédric Hémon (University of Rennes, France)",  
            "Jean-Louis Dillenseger (University of Rennes, France)",
        ]
        self.parent.helpText = _(
            """"""
        )
        self.parent.acknowledgementText = _(
            """"""
        )

def resourcePath(filename):
    """Return the absolute path of the module ``Resources`` directory."""
    scriptedModulesPath = os.path.dirname(slicer.modules.impactreg.path)
    return os.path.join(scriptedModulesPath, "Resources", filename)

def resourceKonfAIPath(filename):
    """Return the absolute path of the module ``Resources`` directory."""
    scriptedModulesPath = os.path.dirname(slicer.modules.konfai.path)
    return os.path.join(scriptedModulesPath, "Resources", filename)

class Preset():

    def __init__(self, repo_id: str, metadata: dict[str, str]) -> None:
        self._display_name = metadata["display_name"]
        self._parameter_maps = []
        for parameter_map in metadata["parameter_maps"]:
            self._parameter_maps.append(hf_hub_download(
                repo_id=repo_id, filename=parameter_map, repo_type="model", revision=None))  # nosec B615
        self._models_names = metadata["models"]
        self._models = []
        
        preprocess_function_filename = metadata["preprocess_function"].split(":")[0] + ".py"
        if preprocess_function_filename in list_repo_files(repo_id, repo_type="model"):
            preprocess_function_path = hf_hub_download(
                    repo_id=repo_id, filename=preprocess_function_filename, repo_type="model", revision=None, force_download=False)  # nosec B615
            
            spec = importlib.util.spec_from_file_location("tmp_module", preprocess_function_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._preprocess_function = getattr(module, metadata["preprocess_function"].split(":")[1])
        else:
            self._preprocess_function = lambda x : x

        self._iterations = int(metadata["iterations"])
        self._short_description = metadata["short_description"]
        self._description = metadata["description"]
    
    def get_display_name(self):
        return self._display_name
    
    def install(self) -> tuple[list[str], list[str]]:
        for model_name in self._models_names:
            self._models.append(hf_hub_download(
                repo_id=model_name.split(":")[0], filename=model_name.split(":")[1], repo_type="model", revision=None))  # nosec B615
        return self._parameter_maps, self._models
    
    def preprocess(self, image: sitk.Image) -> sitk.Image:
        return self._preprocess_function(image)

    def get_models_name(self):
        return [model_name.split(":")[1] for model_name in self._models_names]
    
    def get_number_of_iterations(self) -> int:
        return self._iterations
    
    def set_device(self, parametermap_path: str, device_str: str):
        device = -1
        if device_str != "None":
            device = int(device_str.split(",")[0])
        
        lines = []
        with open(parametermap_path, "r") as f:
            for line in f:
                if line.strip().startswith("(ImpactGPU"):
                    lines.append(f"(ImpactGPU {device})\n")
                else:
                    lines.append(line)

        with open(parametermap_path, "w") as f:
            f.writelines(lines)

    def get_description(self) -> str:
        return self._description
    
    def get_short_description(self) -> str:
        return self._short_description
    
    def get_preset_dir(self) -> Path:
        return Path(self._parameter_maps[0])
    
class ElastixImpactWidget(AppTemplateWidget):

    def __init__(self, name: str, repo_id: str):
        super().__init__(name, slicer.util.loadUI(resourcePath("UI/ElastixImpactReg.ui")))
        self.repo_id = repo_id

        self.evaluationPanel = KonfAIMetricsPanel()
        self.ui.withRefMetricsPlaceholder.layout().addWidget(self.evaluationPanel)
        self.uncertaintyPanel = KonfAIMetricsPanel()
        self.ui.noRefMetricsPlaceholder.layout().addWidget(self.uncertaintyPanel)

        self._description_expanded = False

        self.ui.fixedVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        self.ui.movingVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        
        self.ui.fixedMaskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        self.ui.movingMaskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        
        self.ui.inputTransformSequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        
        self.ui.fixedImageEvaluationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        self.ui.movingImageEvaluationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        self.ui.fixedSegEvaluationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        self.ui.movingSegEvaluationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        self.ui.fixedFidEvaluationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        self.ui.movingFidEvaluationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        
        self.ui.inputTransformSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_GUI)
        
        self.ui.toggleDescriptionButton.clicked.connect(self.on_toggle_description)
        self.ui.runRegistrationButton.clicked.connect(self.on_run_registration_button)
        self.ui.runEvaluationButton.clicked.connect(self.on_run_evaluation_button)

        iconPath = resourceKonfAIPath("Icons/gear.png")
        self.ui.presetButton.setIcon(QIcon(iconPath))
        self.ui.presetButton.setIconSize(QSize(18, 18))
        self.ui.presetButton.clicked.connect(self.on_open_config)

        self.elastix_bin = ElastixImpactWidget.get_elastix_bin()


        preset_database_path = hf_hub_download(
            repo_id=repo_id, filename="PresetDatabase.json", repo_type="model", revision=None)  # nosec B615
        
        with open(preset_database_path, encoding="utf-8") as f:
            preset_database = json.load(f)
        
        for preset_metadata in preset_database["presets"]:
            preset = Preset(repo_id, preset_metadata)
            self.ui.parameterMapPresetComboBox.addItem(preset.get_display_name(), preset)


        self.ui.parameterMapPresetComboBox.currentIndexChanged.connect(self.on_preset_selected)

    def on_run_evaluation_button(self):
        self.evaluationPanel.clearImagesList()
        self.uncertaintyPanel.clearImagesList()
        self.on_run_button(self.evaluation if self.ui.qaTabWidget.currentWidget().name == "withRefTab" else self.uncertainty)

    def evaluation(self):

        args = [
            "eval",
            f"{self.repo_id}:ImpactReg",
            "-i", 
            "Volume.mha",
            "--gt",
            "Reference.mha",
            "-o",
            "Evaluation",
        ]
        if self._parameterNode.GetParameter("Device") != "None":
            args += ["--gpu", self._parameterNode.GetParameter("Device")]
        else:
            args += ["--cpu", "1"]
        
        def on_end_function() -> None:
            if self.process.exitStatus() != QProcess.NormalExit:
                self.set_running(False)
                return
            try:
                statistics = Statistics((self._work_dir / "Evaluation").rglob("*.json").__next__())
                self.evaluationPanel.setMetrics(statistics.read()) 
                self.evaluationPanel.refreshImagesList(Path((self._work_dir / "Evaluation").rglob("*.mha").__next__().parent))
                self._update_logs("Processing finished.")
                self.set_running(False)
            finally:
                self.set_running(False)
        self.process.run("konfai-apps", self._work_dir, args, on_end_function)
    
    def uncertainty(self):
        args = [
            "uncertainty",
            f"{self.repo_id}:ImpactReg",
            "-i", 
            "Volume.mha",
            "-o",
            "Uncertainty",
        ]

        if self._parameterNode.GetParameter("Device") != "None":
            args += ["--gpu", self._parameterNode.GetParameter("Device")]
        else:
            args += ["--cpu", "1"]
        
        def on_end_function() -> None:
            if self.process.exitStatus() != QProcess.NormalExit:
                self.set_running(False)
                return
            try:
                statistics = Statistics((self._work_dir / "Uncertainty").rglob("*.json").__next__())
                self.uncertaintyPanel.setMetrics(statistics.read()) 
                self.uncertaintyPanel.refreshImagesList(Path((self._work_dir / "Uncertainty").rglob("*.mha").__next__().parent))
                self._update_logs("Processing finished.")
            finally:
                self.set_running(False)

        self.process.run("konfai-apps", self._work_dir, args, on_end_function)


    def setup_preset_chips(self, presets: list[str]):
        combo = self.ui.parameterMapPresetComboBox
        layout = self.ui.selectedPresetsWidget.layout()

        def chip_exists(text):
            for i in range(layout.count()):
                item = layout.itemAt(i)
                w = item.widget()
                if isinstance(w, QPushButton) and w.text == text:
                    return True
            return False

        def insert_before_spacer(widget):
            insert_index = layout.count()
            for i in range(layout.count()):
                if layout.itemAt(i).spacerItem() is not None:
                    insert_index = i
                    break
            layout.insertWidget(insert_index, widget)

        def add_chip(text):
            if not text or chip_exists(text):
                return

            btn = QPushButton(text)
            btn.flat = True
            btn.toolTip = f"Click to remove preset '{text}'"
            btn.minimumHeight = 20
            btn.maximumHeight = 24

            btn.styleSheet = """
                QPushButton {
                    border: 1px solid #888;
                    border-radius: 10px;
                    padding: 2px 6px;
                    background-color: #e0e0e0;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
            """

            def remove_chip():
                layout.removeWidget(btn)
                btn.deleteLater()
                self._parameterNode.SetParameter(f"{self.name}/Presets", ",".join(self.get_selected_presets()))

            btn.clicked.connect(remove_chip)

            insert_before_spacer(btn)
            self._parameterNode.SetParameter(f"{self.name}/Presets", ",".join(self.get_selected_presets()))

        def on_preset_activated(index):
            add_chip(combo.itemText(index))

        
        combo.connect('activated(int)', on_preset_activated)
        
        for preset in presets:
            add_chip(preset)

    def get_selected_presets(self):
        layout = self.ui.selectedPresetsWidget.layout()
        presets = []
        for i in range(layout.count()):
            item = layout.itemAt(i)
            w = item.widget()
            if isinstance(w, QPushButton):
                presets.append(w.text)
        return presets

    def on_open_config(self):
        """
        Open configuration file when user clicks "Open config" button.
        """
        preset_dir = self.ui.parameterMapPresetComboBox.currentData.get_preset_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(preset_dir.parent))

    def on_preset_selected(self):
        preset: Preset = self.ui.parameterMapPresetComboBox.currentData
        self.ui.removePresetButton.setEnabled(False)
        self._description_expanded = False
        self.on_toggle_description()
        self._parameterNode.SetParameter(f"{self.name}/Preset", str(self.ui.parameterMapPresetComboBox.currentIndex))

    def on_toggle_description(self):
        preset: Preset = self.ui.parameterMapPresetComboBox.currentData
        if self._description_expanded:
            self.ui.presetDescriptionLabel.setText(preset.get_description())
            self.ui.toggleDescriptionButton.setText("Less ▲")
        else:
            self.ui.presetDescriptionLabel.setText(preset.get_short_description())
            self.ui.toggleDescriptionButton.setText("More ▼")
        self._description_expanded = not self._description_expanded

    def app_setup(self, update_logs, update_progress, parameterNode):
        self._update_logs = update_logs
        self._update_progress = update_progress
        self._parameterNode = parameterNode
        self.process = ElastixProcess(update_logs, update_progress)

    def initialize_parameter_node(self):
        self._parameterNode.SetParameter(f"{self.name}/Preset", str(0))


    def initialize_GUI_from_parameter_node(self):
        self.ui.fixedVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference(f"{self.name}/FixedVolume"))
        self.ui.movingVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference(f"{self.name}/MovingVolume"))
        
        self.ui.fixedMaskSelector.setCurrentNode(self._parameterNode.GetNodeReference(f"{self.name}/FixedVolume"))
        self.ui.movingMaskSelector.setCurrentNode(self._parameterNode.GetNodeReference(f"{self.name}/FixedVolume"))
        
        self.ui.inputTransformSequenceSelector.setCurrentNode(self._parameterNode.GetNodeReference(f"{self.name}/TransformSequence"))

        self.ui.fixedImageEvaluationSelector.setCurrentNode(self._parameterNode.GetNodeReference(f"{self.name}/FixedImageEvaluation"))
        self.ui.movingImageEvaluationSelector.setCurrentNode(self._parameterNode.GetNodeReference(f"{self.name}/MovingImageEvaluation"))
        
        self.ui.fixedSegEvaluationSelector.setCurrentNode(self._parameterNode.GetNodeReference(f"{self.name}/FixedSegEvaluation"))
        self.ui.movingSegEvaluationSelector.setCurrentNode(self._parameterNode.GetNodeReference(f"{self.name}/MovingSegEvaluation"))
        
        self.ui.fixedFidEvaluationSelector.setCurrentNode(self._parameterNode.GetNodeReference(f"{self.name}/FixedFidEvaluation"))
        self.ui.movingFidEvaluationSelector.setCurrentNode(self._parameterNode.GetNodeReference(f"{self.name}/MovingFidEvaluation"))
        

        self.ui.inputTransformSelector.setCurrentNode(self._parameterNode.GetNodeReference(f"{self.name}/TransformEvaluation"))
        
        self.setup_preset_chips(self._parameterNode.GetParameter(f"{self.name}/Presets").split(","))
        
        self.ui.parameterMapPresetComboBox.setCurrentIndex(int(self._parameterNode.GetParameter(f"{self.name}/Preset")))

    def enter(self):
        super().enter()
        self.on_preset_selected()

    def update_GUI_from_parameter_node(self):
        fixedVolume = self._parameterNode.GetNodeReference(f"{self.name}/FixedVolume")
        movingVolume = self._parameterNode.GetNodeReference(f"{self.name}/MovingVolume")
        presets = self._parameterNode.GetParameter(f"{self.name}/Presets")
        if fixedVolume and fixedVolume.GetImageData() and movingVolume and movingVolume.GetImageData() and len(presets):
            self.ui.runRegistrationButton.toolTip = _("Start evaluation")
            self.ui.runRegistrationButton.enabled = True
        else:
            self.ui.runRegistrationButton.toolTip = _("Select input and reference volumes")
            self.ui.runRegistrationButton.enabled = False
        
        if not self.is_running():
            self.ui.runRegistrationButton.text = "Run"
            self.ui.runEvaluationButton.text = "Run"            
        else:
            self.ui.runRegistrationButton.text = "Stop"
            self.ui.runEvaluationButton.text = "Stop"
        
        fixedImageEvaluation = self._parameterNode.GetNodeReference(f"{self.name}/FixedImageEvaluation")
        movingImageEvaluation = self._parameterNode.GetNodeReference(f"{self.name}/MovingImageEvaluation")

        fixedSegEvaluation = self._parameterNode.GetNodeReference(f"{self.name}/FixedSegEvaluation")
        movingSegEvaluation = self._parameterNode.GetNodeReference(f"{self.name}/MovingSegEvaluation")

        fixedFidEvaluation = self._parameterNode.GetNodeReference(f"{self.name}/FixedFidEvaluation")
        movingFidEvaluation = self._parameterNode.GetNodeReference(f"{self.name}/MovingFidEvaluation")

        transformEvaluation = self._parameterNode.GetNodeReference(f"{self.name}/TransformEvaluation")
        transformSequence = self._parameterNode.GetNodeReference(f"{self.name}/TransformSequence")

        if self.ui.qaTabWidget.currentWidget().name == "withRefTab":
            if (fixedImageEvaluation and fixedImageEvaluation.GetImageData() and movingImageEvaluation and movingImageEvaluation.GetImageData()) or \
                (fixedSegEvaluation and fixedSegEvaluation.GetImageData() and movingSegEvaluation and movingSegEvaluation.GetImageData()) or \
                (fixedFidEvaluation and fixedFidEvaluation.GetImageData() and movingFidEvaluation and movingFidEvaluation.GetImageData()) and transformEvaluation:
                self.ui.runEvaluationButton.toolTip = _("Start evaluation")
                self.ui.runEvaluationButton.enabled = True
            else:
                self.ui.runEvaluationButton.toolTip = _("Select fixed and moving and transform")
                self.ui.runEvaluationButton.enabled = False
        else:
            if transformSequence and transformSequence.GetNumberOfDataNodes() > 1:
                self.ui.runEvaluationButton.toolTip = _("Start uncertainty estimation")
                self.ui.runEvaluationButton.enabled = True
            else:
                self.ui.runEvaluationButton.toolTip = _("Select input volume")
                self.ui.runEvaluationButton.enabled = False

    def update_parameter_node_from_GUI(self, caller=None, event=None):
        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch
        self._parameterNode.SetNodeReferenceID(f"{self.name}/FixedVolume", self.ui.fixedVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID(f"{self.name}/MovingVolume", self.ui.movingVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID(f"{self.name}/FixedMask", self.ui.fixedMaskSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID(f"{self.name}/MovingMask", self.ui.movingMaskSelector.currentNodeID)

        self._parameterNode.SetNodeReferenceID(f"{self.name}/OutputTransform", self.ui.outputTransformSelector.currentNodeID)

        self._parameterNode.SetNodeReferenceID(f"{self.name}/FixedImageEvaluation", self.ui.fixedImageEvaluationSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID(f"{self.name}/MovingImageEvaluation", self.ui.movingImageEvaluationSelector.currentNodeID)
        
        self._parameterNode.SetNodeReferenceID(f"{self.name}/FixedSegEvaluation", self.ui.fixedSegEvaluationSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID(f"{self.name}/MovingSegEvaluation", self.ui.movingSegEvaluationSelector.currentNodeID)
        
        self._parameterNode.SetNodeReferenceID(f"{self.name}/FixedFidEvaluation", self.ui.fixedFidEvaluationSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID(f"{self.name}/MovingFidEvaluation", self.ui.movingFidEvaluationSelector.currentNodeID)
        

        self._parameterNode.SetNodeReferenceID(f"{self.name}/TransformEvaluation", self.ui.inputTransformSelector.currentNodeID)

        self._parameterNode.SetNodeReferenceID(f"{self.name}/TransformSequence", self.ui.inputTransformSequenceSelector.currentNodeID)

        
        self._parameterNode.EndModify(wasModified)

    def on_run_registration_button(self):
        self.on_run_button(self.registration)

    @staticmethod
    def get_elastix_bin() -> Path:
        script_path = os.path.dirname(os.path.abspath(__file__))
        elastix_bin_dir_candidates = [
            # install tree
            Path(script_path) / "..",
            Path(script_path) / "../../../bin",
            # build tree
            Path(script_path) /  "../../../../bin",
            Path(script_path) /  "../../../../bin/Release",
            Path(script_path) /  "../../../../bin/Debug",
            Path(script_path) /  "../../../../bin/RelWithDebInfo",
            Path(script_path) /  "../../../../bin/MinSizeRel"]

        for elastix_bin_dir_candidate in elastix_bin_dir_candidates:
            executable = elastix_bin_dir_candidate / ("elastix.exe" if platform.system() == "Windows" else "elastix")
            if executable.exists():
                return executable

        raise ValueError('Elastix not found')

    def next_registration(self, presets: list[Preset], args_init: list[str], fixed_image_node, moving_image_node, transforms: list[sitk.Transform] = []) -> None:
        args = args_init.copy()
        preset = presets.pop(0)

        sitk.WriteImage(preset.preprocess(sitkUtils.PullVolumeFromSlicer(fixed_image_node)), str(self._work_dir / "FixedImage.mha"))
        sitk.WriteImage(preset.preprocess(sitkUtils.PullVolumeFromSlicer(moving_image_node)), str(self._work_dir / "MovingImage.mha"))


        parameter_maps_path, models_path = preset.install()

        for f in self._work_dir.iterdir():
            if f.suffix != ".mha":
                if f.is_file():
                    f.unlink()
                else:
                    shutil.rmtree(f)
                    
        for model_path, model_name in zip(models_path, preset.get_models_name()):
            link_path = (self._work_dir / model_name)
            if not link_path.exists():
                link_path.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(model_path, link_path)

        for parameter_map_path in parameter_maps_path:
            copy_of_parameter_map_path = str(self._work_dir / os.path.basename(parameter_map_path))
            shutil.copy2(parameter_map_path, copy_of_parameter_map_path)
            preset.set_device(copy_of_parameter_map_path, self._parameterNode.GetParameter("Device"))  
            args += ["-p", copy_of_parameter_map_path]

        def on_end_elastix() -> None:
            if self.process.exitStatus() != QProcess.NormalExit:
                self.set_running(False)
                return

            files = list(self._work_dir.glob("TransformParameters.*-Composite.itk.txt"))

            if not files:
                raise FileNotFoundError("No transform file could be found.")

            def get_index(path):
                name = path.name
                return int(name.split(".")[1].split("-")[0])

            latest_file = max(files, key=get_index)
            transforms.append(sitk.ReadTransform(str(latest_file)))

            tmpNode = slicer.util.loadTransform(str(latest_file))
            transformNode = slicer.vtkMRMLTransformNode()
            transformNode.SetScene(slicer.mrmlScene)
            transformNode.SetName(f"ElastixTransform_{len(transforms)}")
            transformNode.SetAddToScene(False)   # KEY TRICK: Node will NOT appear in the Data module
            transformNode.Copy(tmpNode)
            
            sequenceNode = self.ui.inputTransformSequenceSelector.currentNode()
            if sequenceNode is None:
                sequenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", "InputTransformSequence")
                self.ui.inputTransformSequenceSelector.setCurrentNode(sequenceNode)

            sequenceNode.SetDataNodeAtValue(transformNode, str(len(transforms)))
            
            slicer.mrmlScene.RemoveNode(tmpNode) 

            if len(presets):
                self.next_registration(presets, args_init, fixed_image_node, moving_image_node, transforms)
            else:
                self.on_end_function(fixed_image_node, moving_image_node, transforms)

        self.process.run(str(self.elastix_bin), self._work_dir, args, on_end_elastix)


    def on_end_function(self, fixed_image_node, moving_image_node, transforms: list[sitk.Transform]) -> None:
        print("ENDDDDDDDDDDDDD")

        browserNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSequenceBrowserNode", "SitkTransformSequenceBrowser"
        )
        sequenceNode = self.ui.inputTransformSequenceSelector.currentNode()
        browserNode.SetAndObserveMasterSequenceNodeID(sequenceNode.GetID())
        
        fixed_image = sitkUtils.PullVolumeFromSlicer(fixed_image_node)

        displacement_fields = []

        for t in transforms:
            field = sitk.TransformToDisplacementField(
                t,
                sitk.sitkVectorFloat64,
                fixed_image.GetSize(),
                fixed_image.GetOrigin(),
                fixed_image.GetSpacing(),
                fixed_image.GetDirection(),
            )
            displacement_fields.append(field)

        arrays = [sitk.GetArrayFromImage(f) for f in displacement_fields]


        avg_array = np.mean(arrays, axis=0)

        avg_field = sitk.GetImageFromArray(avg_array)
        avg_field.CopyInformation(fixed_image)

        avg_transform = sitk.DisplacementFieldTransform(avg_field)
        
        avg_tranform_path = str(self._work_dir / "AverageTransform.h5")
        sitk.WriteTransform(avg_transform, avg_tranform_path)
        
        avg_transform_node = slicer.util.loadTransform(avg_tranform_path)
        avg_transform_node.SetName("AverageTransform")

        output_transform_node = self.ui.outputTransformSelector.currentNode()
        if output_transform_node is None:
            output_transform_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLTransformNode", "OutputTransform"
            )
            self.ui.outputTransformSelector.setCurrentNode(output_transform_node)

        output_transform_node.Copy(avg_transform_node)
        slicer.mrmlScene.RemoveNode(avg_transform_node)

        warpedVolumeNode = slicer.modules.volumes.logic().CloneVolume(
            slicer.mrmlScene, moving_image_node, moving_image_node.GetName() + "_warped"
        )
        warpedVolumeNode.SetAndObserveTransformNodeID(output_transform_node.GetID())
        #slicer.vtkSlicerTransformLogic().hardenTransform(warpedVolumeNode)
        
        slicer.util.setSliceViewerLayers(background=fixed_image_node, foreground=warpedVolumeNode)
        
        self.set_running(False)

    def registration(self) -> None:        
        args_init = [
            "-f",
            "FixedImage.mha",
            "-m",
            "MovingImage.mha",
            "-out",
            ".",
        ]

        fixed_mask_node = self.ui.fixedMaskSelector.currentNode()
        moving_mask_node = self.ui.movingMaskSelector.currentNode()
        
        if fixed_mask_node and fixed_mask_node.GetImageData():
            sitk.WriteImage(sitkUtils.PullVolumeFromSlicer(fixed_mask_node), str(self._work_dir / "FixedMask.mha"))
            args_init += ["-fMask", "FixedMask.mha"]

        if moving_mask_node and moving_mask_node.GetImageData():
            sitk.WriteImage(sitkUtils.PullVolumeFromSlicer(moving_mask_node), str(self._work_dir / "MovingMask.mha"))
            args_init += ["-mMask", "MovingMask.mha"]

        selected_presets = self.get_selected_presets()
        combo = self.ui.parameterMapPresetComboBox
        presets = []
        for i in range(combo.count):
            if combo.itemText(i) in selected_presets:
                presets.append(combo.itemData(i))

        total_it = 0
        for preset in presets:
            total_it += preset.get_number_of_iterations()
        self.process.set_total_iterations(total_it)
    
        self.next_registration(presets, args_init.copy(), self.ui.fixedVolumeSelector.currentNode(), self.ui.movingVolumeSelector.currentNode())
        

class AdamImpactWidget(AppTemplateWidget):

    def __init__(self, name: str):
        super().__init__(name, slicer.util.loadUI(resourcePath("UI/ElastixImpactReg.ui")))

class ImpactRegWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        self.konfai_core = KonfAICoreWidget("Impact Reg")
        self.konfai_core.register_apps([ElastixImpactWidget("Elastix", "VBoussot/ImpactReg")])

        self.layout.addWidget(self.konfai_core)
        
    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        pass

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        pass

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        pass