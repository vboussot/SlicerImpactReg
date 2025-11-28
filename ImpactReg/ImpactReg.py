import importlib.util
import json
import os
import platform
import re
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813
import sitkUtils
import slicer
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from KonfAI import AppTemplateWidget, KonfAICoreWidget, KonfAIMetricsPanel, Process
from konfai.evaluator import Statistics
from qt import QDesktopServices, QIcon, QProcess, QPushButton, QSize, QUrl, QWidget
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget
from slicer.util import VTKObservationMixin


class ElastixProcess(Process):

    def __init__(self, _update_logs, _update_progress):
        super().__init__(_update_logs, _update_progress)
        self._total_iterations = 0

    def on_stdout_ready(self):
        line = self.readAllStandardOutput().data().decode().strip()
        if line:
            line = line.replace("\r\n", "\n").split("\r")[-1]
            self._update_logs(line)
            is_it = False
            for sub_line in line.split("\n"):
                if re.match(r"^\d+", sub_line):
                    parts = re.split(r"\s+", sub_line)
                    try:
                        time_ms = float(parts[-1])
                        is_it = True
                        self._it += 1
                    except ValueError:
                        continue
            if is_it:
                self._update_progress(int(self._it / self._total_iterations * 100), f"{time_ms:.2f} ms")

    def on_stderr_ready(self) -> None:
        line = self.readAllStandardError().data().decode().strip()
        if line:
            line = line.replace("\r\n", "\n").split("\r")[-1]
            self._update_logs(line)
            match = re.search(r"(\d+)%", line)
            if match:
                percent = int(match.group(1))
                self._update_progress(percent, "")

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
        self.parent.helpText = _("""""")
        self.parent.acknowledgementText = _("""""")


def resource_path(filename):
    """Return the absolute path of the module ``Resources`` directory."""
    scripted_modules_path = os.path.dirname(slicer.modules.impactreg.path)
    return os.path.join(scripted_modules_path, "Resources", filename)


def resource_konfai_path(filename):
    """Return the absolute path of the module ``Resources`` directory."""
    scripted_modules_path = os.path.dirname(slicer.modules.konfai.path)
    return os.path.join(scripted_modules_path, "Resources", filename)


class Preset:

    def __init__(self, repo_id: str, metadata: dict[str, str]) -> None:
        self._display_name = metadata["display_name"]
        self._parameter_maps = []
        for parameter_map in metadata["parameter_maps"]:
            self._parameter_maps.append(
                hf_hub_download(repo_id=repo_id, filename=parameter_map, repo_type="model", revision=None)
            )  # nosec B615
        self._models_names = metadata["models"]
        self._models: list[str] = []

        preprocess_function_filename = metadata["preprocess_function"].split(":")[0] + ".py"
        if preprocess_function_filename in list_repo_files(repo_id, repo_type="model"):
            preprocess_function_path = hf_hub_download(
                repo_id=repo_id,
                filename=preprocess_function_filename,
                repo_type="model",
                revision=None,
                force_download=False,
            )  # nosec B615

            spec = importlib.util.spec_from_file_location("tmp_module", preprocess_function_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load preprocess function from {preprocess_function_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._preprocess_function = getattr(module, metadata["preprocess_function"].split(":")[1])
        else:
            self._preprocess_function = lambda x: x

        self._iterations = int(metadata["iterations"])
        self._short_description = metadata["short_description"]
        self._description = metadata["description"]

    def get_display_name(self):
        return self._display_name

    def install(self) -> tuple[list[str], list[str]]:
        for model_name in self._models_names:
            self._models.append(
                hf_hub_download(
                    repo_id=model_name.split(":")[0],
                    filename=model_name.split(":")[1],
                    repo_type="model",
                    revision=None,
                )
            )  # nosec B615
        return self._parameter_maps, self._models

    def preprocess(self, image: sitk.Image) -> sitk.Image:
        return self._preprocess_function(image)

    def get_models_name(self):
        return [model_name.split(":")[1] for model_name in self._models_names]

    def get_number_of_iterations(self) -> int:
        return self._iterations

    def set_device(self, parametermap_path: str, device_str: str | None):
        device = -1
        if device_str:
            device = int(device_str.split(",")[0])

        lines = []
        with open(parametermap_path) as f:
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
        super().__init__(name, slicer.util.loadUI(resource_path("UI/ElastixImpactReg.ui")))
        self.repo_id = repo_id
        self._elastix_bin: None | Path = None

        self.evaluation_panel = KonfAIMetricsPanel()
        self.ui.withRefMetricsPlaceholder.layout().addWidget(self.evaluation_panel)
        self.uncertainty_panel = KonfAIMetricsPanel()
        self.ui.noRefMetricsPlaceholder.layout().addWidget(self.uncertainty_panel)

        self._description_expanded = False

        self.ui.fixedVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)
        self.ui.movingVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)

        self.ui.fixedMaskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)
        self.ui.movingMaskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)

        self.ui.inputTransformSequenceSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui
        )

        self.ui.fixedImageEvaluationSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui
        )
        self.ui.movingImageEvaluationSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui
        )
        self.ui.fixedSegEvaluationSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui
        )
        self.ui.movingSegEvaluationSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui
        )
        self.ui.fixedFidEvaluationSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui
        )
        self.ui.movingFidEvaluationSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui
        )

        self.ui.referenceMaskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)
        self.ui.inputTransformSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)

        self.ui.toggleDescriptionButton.clicked.connect(self.on_toggle_description)
        self.ui.runRegistrationButton.clicked.connect(self.on_run_registration_button)
        self.ui.runEvaluationButton.clicked.connect(self.on_run_evaluation_button)

        self.ui.qaTabWidget.currentChanged.connect(self.on_tab_changed)

        icon_path = resource_konfai_path("Icons/gear.png")
        self.ui.presetButton.setIcon(QIcon(icon_path))
        self.ui.presetButton.setIconSize(QSize(18, 18))
        self.ui.presetButton.clicked.connect(self.on_open_config)

        preset_database_path = hf_hub_download(
            repo_id=repo_id, filename="PresetDatabase.json", repo_type="model", revision=None
        )  # nosec B615

        with open(preset_database_path, encoding="utf-8") as f:
            preset_database = json.load(f)

        for preset_metadata in preset_database["presets"]:
            preset = Preset(repo_id, preset_metadata)
            self.ui.parameterMapPresetComboBox.addItem(preset.get_display_name(), preset)

        self.ui.parameterMapPresetComboBox.currentIndexChanged.connect(self.on_preset_selected)

    def on_tab_changed(self) -> None:
        """
        Update GUI state when the user switches between QA tabs.

        Ensures that button enabling/disabling is consistent with the current tab.
        """
        self.update_gui_from_parameter_node()

    def on_run_evaluation_button(self):
        self.evaluation_panel.clear_images_list()
        self.uncertainty_panel.clear_images_list()
        self.on_run_button(
            self.evaluation if self.ui.qaTabWidget.currentWidget().name == "withRefTab" else self.uncertainty
        )

    def next_evaluation(self, args_list: list[list[str]]):
        args = args_list.pop(0)
        if self.get_device():
            args += ["--gpu", self.get_device()]
        else:
            args += ["--cpu", "1"]

        def on_end_evaluation(args: list[list[str]]) -> None:
            if self.process.exitStatus() != QProcess.NormalExit:
                self.set_running(False)
                return
            try:
                statistics = Statistics((self._work_dir / "Evaluation").rglob("*.json").__next__())
                self.evaluation_panel.set_metrics(statistics.read())
                self.evaluation_panel.refresh_images_list(
                    Path((self._work_dir / "Evaluation").rglob("*.mha").__next__().parent)
                )
                self._update_logs("Processing finished.")
                if len(args_list) == 0:
                    self.set_running(False)
                else:
                    self.next_evaluation(args_list)
            except Exception as e:
                print(e)
                self.set_running(False)

        self.process.run("konfai-apps", self._work_dir, args, on_end_evaluation)

    def evaluation(self):
        self.evaluation_panel.clear_metrics()
        fixed_image_evaluation = self.ui.fixedImageEvaluationSelector.currentNode()
        moving_image_evaluation = self.ui.movingImageEvaluationSelector.currentNode()

        fixed_seg_evaluation = self.ui.fixedSegEvaluationSelector.currentNode()
        moving_seg_evaluation = self.ui.movingSegEvaluationSelector.currentNode()

        fixed_fid_evaluation = self.ui.fixedFidEvaluationSelector.currentNode()
        moving_fid_evaluation = self.ui.movingFidEvaluationSelector.currentNode()

        mask_evaluation = self.ui.referenceMaskSelector.currentNode()
        transform_evaluation = self.ui.inputTransformSelector.currentNode()

        args_list = []

        if (
            fixed_image_evaluation
            and fixed_image_evaluation.GetImageData()
            and moving_image_evaluation
            and moving_image_evaluation.GetImageData()
        ):
            volume_storage_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
            volume_storage_node.SetFileName(str(self._work_dir / "FixedImage.mha"))
            volume_storage_node.UseCompressionOff()
            volume_storage_node.WriteData(fixed_image_evaluation)
            volume_storage_node.UnRegister(None)

            warped_volume_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", moving_image_evaluation.GetName() + "_warped"
            )

            params = {
                "inputVolume": moving_image_evaluation.GetID(),
                "referenceVolume": fixed_image_evaluation.GetID(),
                "outputVolume": warped_volume_node.GetID(),
                "interpolationType": "linear",
                "transformationFile": transform_evaluation.GetID(),
            }

            slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, params)

            volume_storage_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
            volume_storage_node.SetFileName(str(self._work_dir / "MovingImage.mha"))
            volume_storage_node.UseCompressionOff()
            volume_storage_node.WriteData(warped_volume_node)
            volume_storage_node.UnRegister(None)
            args_list.append(
                [
                    "eval",
                    f"{self.repo_id}:ImpactReg",
                    "-i",
                    "FixedImage.mha",
                    "--gt",
                    "MovingImage.mha",
                    "-o",
                    "Evaluation",
                    "--evaluation_file",
                    "Evaluation_with_images.yml",
                ]
            )

        if (
            fixed_seg_evaluation
            and fixed_seg_evaluation.GetImageData()
            and moving_seg_evaluation
            and moving_seg_evaluation.GetImageData()
        ):
            volume_storage_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
            volume_storage_node.SetFileName(str(self._work_dir / "FixedSeg.mha"))
            volume_storage_node.UseCompressionOff()
            volume_storage_node.WriteData(fixed_seg_evaluation)
            volume_storage_node.UnRegister(None)

            warped_seg_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", moving_seg_evaluation.GetName() + "_warped"
            )

            params = {
                "inputVolume": moving_seg_evaluation.GetID(),
                "referenceVolume": fixed_seg_evaluation.GetID(),
                "outputVolume": warped_seg_node.GetID(),
                "interpolationType": "nn",
                "transformationFile": transform_evaluation.GetID(),
            }

            slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, params)

            volume_storage_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
            volume_storage_node.SetFileName(str(self._work_dir / "MovingSeg.mha"))
            volume_storage_node.UseCompressionOff()
            volume_storage_node.WriteData(warped_seg_node)
            volume_storage_node.UnRegister(None)

            args_list.append(
                [
                    "eval",
                    f"{self.repo_id}:ImpactReg",
                    "-i",
                    "FixedSeg.mha",
                    "--gt",
                    "MovingSeg.mha",
                    "-o",
                    "Evaluation",
                    "--evaluation_file",
                    "Evaluation_with_seg.yml",
                ]
            )

        if (
            fixed_fid_evaluation
            and fixed_fid_evaluation.GetNumberOfControlPoints()
            and moving_fid_evaluation
            and moving_fid_evaluation.GetNumberOfControlPoints()
        ):

            volume_storage_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLMarkupsFiducialStorageNode")
            volume_storage_node.SetFileName(str(self._work_dir / "FixedFid.fcsv"))
            volume_storage_node.WriteData(fixed_fid_evaluation)
            volume_storage_node.UnRegister(None)

            warped_landmarks = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", moving_fid_evaluation.GetName() + "_warped"
            )
            warped_landmarks.Copy(moving_fid_evaluation)
            warped_landmarks.SetName(moving_fid_evaluation.GetName() + "_warped")

            warped_landmarks.SetAndObserveDisplayNodeID(None)
            warped_landmarks.CreateDefaultDisplayNodes()

            warped_landmarks.SetAndObserveTransformNodeID(transform_evaluation.GetID())
            slicer.vtkSlicerTransformLogic().hardenTransform(warped_landmarks)

            volume_storage_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLMarkupsFiducialStorageNode")
            volume_storage_node.SetFileName(str(self._work_dir / "MovingFid.fcsv"))
            volume_storage_node.WriteData(warped_landmarks)
            volume_storage_node.UnRegister(None)

            args_list.append(
                [
                    "eval",
                    f"{self.repo_id}:ImpactReg",
                    "-i",
                    "FixedFid.fcsv",
                    "--gt",
                    "MovingFid.fcsv",
                    "-o",
                    "Evaluation",
                    "--evaluation_file",
                    "Evaluation_with_fid.yml",
                ]
            )
        if mask_evaluation and mask_evaluation.GetImageData():
            volume_storage_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
            volume_storage_node.SetFileName(str(self._work_dir / "Mask.mha"))
            volume_storage_node.UseCompressionOff()
            volume_storage_node.WriteData(mask_evaluation)
            volume_storage_node.UnRegister(None)

            for args in args_list:
                args += ["--mask", "Mask.mha"]
        self.next_evaluation(args_list)

    def uncertainty(self):
        """self.uncertainty_panel.clear_metrics()
        transform_sequence_node = self.ui.inputTransformSequenceSelector.currentNode()

        composite = sitk.CompositeTransform(3)

        transform_path_tmp = self._work_dir / "Transforms"
        transform_path_tmp.mkdir(parents=True, exist_ok=True)

        for i in range(transform_sequence_node.GetNumberOfDataNodes()):
            tnode = transform_sequence_node.GetNthDataNode(i)
            sitk_transform = sitkUtils.PullTransformFromSlicer(tnode)
            # composite.AddTransform(sitk.ReadTransform(str(transform_filename_tmp)))

        # sitk.WriteTransform(composite, str(self._work_dir / "Transform.itk.txt"))


        args = [
            "uncertainty",
            f"{self.repo_id}:ImpactReg",
            "-i",
            "Volume.mha",
            "-o",
            "Uncertainty",
        ]

        if self.get_device():
            args += ["--gpu", self.get_device()]
        else:
            args += ["--cpu", "1"]

        def on_end_function() -> None:
            if self.process.exitStatus() != QProcess.NormalExit:
                self.set_running(False)
                return
            try:
                statistics = Statistics((self._work_dir / "Uncertainty").rglob("*.json").__next__())
                self.uncertainty_panel.set_metrics(statistics.read())
                self.uncertainty_panel.refresh_images_list(
                    Path((self._work_dir / "Uncertainty").rglob("*.mha").__next__().parent)
                )
                self._update_logs("Processing finished.")
            finally:
                self.set_running(False)

        self.process.run("konfai-apps", self._work_dir, args, on_end_function)"""
        pass

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
                self.set_parameter("Presets", ",".join(self.get_selected_presets()))

            btn.clicked.connect(remove_chip)

            insert_before_spacer(btn)
            self.set_parameter("Presets", ",".join(self.get_selected_presets()))

        def on_preset_activated(index):
            add_chip(combo.itemText(index))

        combo.connect("activated(int)", on_preset_activated)

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
        self.ui.removePresetButton.setEnabled(False)
        self._description_expanded = False
        self.on_toggle_description()
        self.set_parameter("Preset", str(self.ui.parameterMapPresetComboBox.currentIndex))

        self.ui.qaTabWidget.setTabEnabled(1, False)

    def on_toggle_description(self):
        preset: Preset = self.ui.parameterMapPresetComboBox.currentData
        if self._description_expanded:
            self.ui.presetDescriptionLabel.setText(preset.get_description())
            self.ui.toggleDescriptionButton.setText("Less ▲")
        else:
            self.ui.presetDescriptionLabel.setText(preset.get_short_description())
            self.ui.toggleDescriptionButton.setText("More ▼")
        self._description_expanded = not self._description_expanded

    def app_setup(self, update_logs, update_progress, parameter_node):
        self._update_logs = update_logs
        self._update_progress = update_progress
        self._parameter_node = parameter_node
        self.process = ElastixProcess(update_logs, update_progress)

    def initialize_parameter_node(self):
        """
        Initialize the parameter node with default values for this app
        (input volume, model ID, ensemble/TTA/MC-dropout parameters).
        """
        self._initialized = False
        if self.get_parameter_node("FixedVolume") is None:
            fixed_volume_node = slicer.mrmlScene.GetNthNodeByClass(0, "vtkMRMLScalarVolumeNode")
            if fixed_volume_node and self._parameter_node is not None:
                self._parameter_node.SetNodeReferenceID(f"{self._name}/FixedVolume", fixed_volume_node.GetID())

        if self.get_parameter_node("MovingVolume") is None:
            second_volume_node = slicer.mrmlScene.GetNthNodeByClass(1, "vtkMRMLScalarVolumeNode")
            if second_volume_node and self._parameter_node is not None:
                self._parameter_node.SetNodeReferenceID(f"{self._name}/MovingVolume", second_volume_node.GetID())

        if not self.get_parameter("Preset"):
            self.set_parameter("Preset", "0")

        self.initialize_gui_from_parameter_node()
        self._initialized = True

    def initialize_gui_from_parameter_node(self):
        """
        Initialize GUI widget values from the parameter node.
        """
        self.ui.fixedVolumeSelector.setCurrentNode(self.get_parameter_node("FixedVolume"))
        self.ui.movingVolumeSelector.setCurrentNode(self.get_parameter_node("MovingVolume"))

        self.ui.fixedMaskSelector.setCurrentNode(self.get_parameter_node("FixedMask"))
        self.ui.movingMaskSelector.setCurrentNode(self.get_parameter_node("MovingMask"))

        self.ui.outputTransformSelector.setCurrentNode(self.get_parameter_node("OutputTransform"))

        self.ui.inputTransformSequenceSelector.setCurrentNode(self.get_parameter_node("TransformSequence"))

        self.ui.fixedImageEvaluationSelector.setCurrentNode(self.get_parameter_node("FixedImageEvaluation"))
        self.ui.movingImageEvaluationSelector.setCurrentNode(self.get_parameter_node("MovingImageEvaluation"))

        self.ui.fixedSegEvaluationSelector.setCurrentNode(self.get_parameter_node("FixedSegEvaluation"))

        self.ui.movingSegEvaluationSelector.setCurrentNode(self.get_parameter_node("MovingSegEvaluation"))

        self.ui.fixedFidEvaluationSelector.setCurrentNode(self.get_parameter_node("FixedFidEvaluation"))
        self.ui.movingFidEvaluationSelector.setCurrentNode(self.get_parameter_node("MovingFidEvaluation"))

        self.ui.referenceMaskSelector.setCurrentNode(self.get_parameter_node("MaskEvaluation"))
        self.ui.inputTransformSelector.setCurrentNode(self.get_parameter_node("TransformEvaluation"))

        self.setup_preset_chips(self.get_parameter("Presets").split(","))

        self.ui.parameterMapPresetComboBox.setCurrentIndex(int(self.get_parameter("Preset")))

    def enter(self):
        super().enter()
        self.on_preset_selected()

    def update_gui_from_parameter_node(self):
        fixed_volume = self.get_parameter_node("FixedVolume")
        moving_volume = self.get_parameter_node("MovingVolume")
        presets = self.get_parameter("Presets")
        if (
            fixed_volume
            and fixed_volume.GetImageData()
            and moving_volume
            and moving_volume.GetImageData()
            and len(presets)
        ):
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

        fixed_image_evaluation = self.get_parameter_node("FixedImageEvaluation")
        moving_image_evaluation = self.get_parameter_node("MovingImageEvaluation")

        fixed_seg_evaluation = self.get_parameter_node("FixedSegEvaluation")
        moving_seg_evaluation = self.get_parameter_node("MovingSegEvaluation")

        fixed_fid_evaluation = self.get_parameter_node("FixedFidEvaluation")
        moving_fid_evaluation = self.get_parameter_node("MovingFidEvaluation")

        transform_evaluation = self.get_parameter_node("TransformEvaluation")
        transform_sequence = self.get_parameter_node("TransformSequence")

        if self.ui.qaTabWidget.currentWidget().name == "withRefTab":
            if (
                (
                    fixed_image_evaluation
                    and fixed_image_evaluation.GetImageData()
                    and moving_image_evaluation
                    and moving_image_evaluation.GetImageData()
                )
                or (
                    fixed_seg_evaluation
                    and fixed_seg_evaluation.GetImageData()
                    and moving_seg_evaluation
                    and moving_seg_evaluation.GetImageData()
                )
                or (
                    fixed_fid_evaluation
                    and fixed_fid_evaluation.GetNumberOfControlPoints() > 0
                    and moving_fid_evaluation
                    and moving_fid_evaluation.GetNumberOfControlPoints() > 0
                )
            ) and transform_evaluation:
                self.ui.runEvaluationButton.toolTip = _("Start evaluation")
                self.ui.runEvaluationButton.enabled = True
            else:
                self.ui.runEvaluationButton.toolTip = _("Select fixed and moving and transform")
                self.ui.runEvaluationButton.enabled = False
        else:
            if transform_sequence and transform_sequence.GetNumberOfDataNodes() > 1:
                self.ui.runEvaluationButton.toolTip = _("Start uncertainty estimation")
                self.ui.runEvaluationButton.enabled = True
            else:
                self.ui.runEvaluationButton.toolTip = _("Select input volume")
                self.ui.runEvaluationButton.enabled = False
        # Suggest an output volume base name derived from input volume name
        if moving_volume:
            self.ui.outputTransformSelector.baseName = _("{volume_name} Transform").format(
                volume_name=moving_volume.GetName()
            )

    def update_parameter_node_from_gui(self, caller=None, event=None):
        if self._parameter_node is None or not self._initialized:
            return
        was_modified = self._parameter_node.StartModify()

        self.set_parameter_node("FixedVolume", self.ui.fixedVolumeSelector.currentNodeID)
        self.set_parameter_node("MovingVolume", self.ui.movingVolumeSelector.currentNodeID)
        self.set_parameter_node("FixedMask", self.ui.fixedMaskSelector.currentNodeID)
        self.set_parameter_node("MovingMask", self.ui.movingMaskSelector.currentNodeID)

        self.set_parameter_node("OutputTransform", self.ui.outputTransformSelector.currentNodeID)

        self.set_parameter_node("FixedImageEvaluation", self.ui.fixedImageEvaluationSelector.currentNodeID)
        self.set_parameter_node("MovingImageEvaluation", self.ui.movingImageEvaluationSelector.currentNodeID)

        self.set_parameter_node("FixedSegEvaluation", self.ui.fixedSegEvaluationSelector.currentNodeID)
        self.set_parameter_node("MovingSegEvaluation", self.ui.movingSegEvaluationSelector.currentNodeID)

        self.set_parameter_node("FixedFidEvaluation", self.ui.fixedFidEvaluationSelector.currentNodeID)
        self.set_parameter_node("MovingFidEvaluation", self.ui.movingFidEvaluationSelector.currentNodeID)

        self.set_parameter_node("MaskEvaluation", self.ui.referenceMaskSelector.currentNodeID)
        self.set_parameter_node("TransformEvaluation", self.ui.inputTransformSelector.currentNodeID)

        self.set_parameter_node("TransformSequence", self.ui.inputTransformSequenceSelector.currentNodeID)

        self._parameter_node.EndModify(was_modified)

    def on_run_registration_button(self):
        self.on_run_button(self.registration)

    def get_elastix_bin(self) -> None:
        file = "elastix-impact-{}-shared-with-deps-{}".format(
            "win64" if platform.system() == "Windows" else "linux", "cu126" if torch.cuda.is_available() else "cpu"
        )
        path = Path(os.path.dirname(os.path.abspath(__file__))) / "Resources" / "bin"
        executable = "elastix.exe" if platform.system() == "Windows" else "elastix"
        matches = list((path / file).rglob(executable))
        if len(matches) > 0:
            if len(matches) > 1:
                print("[WARNING] Multiple elastix binaries found, using the first one:")
                for m in matches:
                    print(f"  - {m}")

            self._elastix_bin = matches[0]
            file_path = Path(self._elastix_bin)
            if platform.system() != "Windows":
                if not os.access(file_path, os.X_OK):
                    print(f"[INFO] Setting execute permission: {file_path}")
                    file_path.chmod(file_path.stat().st_mode | 0o111)
            self.registration()
            return

        def on_en_function():
            if self.process.exitStatus() != QProcess.NormalExit:
                self.set_running(False)
                return
            self._elastix_bin = path / executable
            if self._elastix_bin.exists():
                self.registration()
            self.set_running(False)

        self.process.run(shutil.which("PythonSlicer"), path, ["Download.py"], on_en_function)

    def next_registration(
        self,
        presets: list[Preset],
        args_init: list[str],
        fixed_image_node,
        moving_image_node,
        transforms: list[sitk.Transform] = [],
    ) -> None:
        args = args_init.copy()
        preset = presets.pop(0)

        sitk.WriteImage(
            preset.preprocess(sitkUtils.PullVolumeFromSlicer(fixed_image_node)), str(self._work_dir / "FixedImage.mha")
        )
        sitk.WriteImage(
            preset.preprocess(sitkUtils.PullVolumeFromSlicer(moving_image_node)),
            str(self._work_dir / "MovingImage.mha"),
        )

        parameter_maps_path, models_path = preset.install()

        for f in self._work_dir.iterdir():
            if f.suffix != ".mha":
                if f.is_file():
                    f.unlink()
                else:
                    shutil.rmtree(f)

        for model_path, model_name in zip(models_path, preset.get_models_name()):
            link_path = self._work_dir / model_name
            if not link_path.exists():
                link_path.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(model_path, link_path)

        for parameter_map_path in parameter_maps_path:
            copy_of_parameter_map_path = str(self._work_dir / os.path.basename(parameter_map_path))
            shutil.copy2(parameter_map_path, copy_of_parameter_map_path)
            preset.set_device(copy_of_parameter_map_path, self.get_device())
            args += ["-p", copy_of_parameter_map_path]

        def on_end_elastix() -> None:
            if self.process.exitStatus() != QProcess.NormalExit:
                self.set_running(False)
                return
            try:
                files = list(self._work_dir.glob("TransformParameters.*-Composite.itk.txt"))

                if not files:
                    raise FileNotFoundError("No transform file could be found.")

                def get_index(path):
                    name = path.name
                    return int(name.split(".")[1].split("-")[0])

                latest_file = max(files, key=get_index)
                transforms.append(sitk.ReadTransform(str(latest_file)))

                tmp_node = slicer.util.loadTransform(str(latest_file))
                tmp_node.SetName(f"ElastixTransform_{len(transforms)-1}")

                sequence_node = self.ui.inputTransformSequenceSelector.currentNode()
                if sequence_node is None:
                    sequence_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", "InputTransformSequence")
                    self.ui.inputTransformSequenceSelector.setCurrentNode(sequence_node)

                sequence_node.SetDataNodeAtValue(tmp_node, str(len(transforms) - 1))

                if len(presets):
                    self.next_registration(presets, args_init, fixed_image_node, moving_image_node, transforms)
                else:
                    self.on_end_function(fixed_image_node, moving_image_node, transforms)
            except Exception as e:
                print(e)
                self.set_running(False)

        self.process.run(self._elastix_bin, self._work_dir, args, on_end_elastix)

    def on_end_function(self, fixed_image_node, moving_image_node, transforms: list[sitk.Transform]) -> None:
        browser_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", "SitkTransformSequenceBrowser")
        sequence_node = self.ui.inputTransformSequenceSelector.currentNode()
        browser_node.SetAndObserveMasterSequenceNodeID(sequence_node.GetID())

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
            output_transform_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "OutputTransform")
            self.ui.outputTransformSelector.setCurrentNode(output_transform_node)

        output_transform_node.Copy(avg_transform_node)
        slicer.mrmlScene.RemoveNode(avg_transform_node)

        warped_volume_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", moving_image_node.GetName() + "_warped"
        )

        params = {
            "inputVolume": moving_image_node.GetID(),
            "referenceVolume": fixed_image_node.GetID(),
            "outputVolume": warped_volume_node.GetID(),
            "interpolationType": "linear",
            "transformationFile": output_transform_node.GetID(),
        }

        slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, params)

        slicer.util.setSliceViewerLayers(background=fixed_image_node, foreground=warped_volume_node)

        self.set_running(False)

    def registration(self) -> None:
        if self._elastix_bin is None:
            self.get_elastix_bin()
            return

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

        sequence_node = self.ui.inputTransformSequenceSelector.currentNode()
        if sequence_node is not None:
            sequence_node.RemoveAllDataNodes()
        self.next_registration(
            presets,
            args_init.copy(),
            self.ui.fixedVolumeSelector.currentNode(),
            self.ui.movingVolumeSelector.currentNode(),
        )


class ImpactRegWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """
    Top-level scripted loadable module widget for SlicerImpactReg.

    This class ties together the Slicer module system with the KonfAICoreWidget,
    which handles actual application logic and GUI.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        super().__init__(parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

    def setup(self) -> None:
        """
        Construct and initialize the KonfAI module GUI.

        This method is called once when the user first opens the module.
        """
        super().setup()

        # Create the core KonfAI widget
        self.konfai_core = KonfAICoreWidget("Impact Reg")

        # Create and register one KonfAI app specialized for registration
        self.konfai_core.register_apps([ElastixImpactWidget("Elastix", "VBoussot/ImpactReg")])

        # Attach the core widget to the Slicer module layout
        self.layout.addWidget(self.konfai_core)

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()
        self.konfai_core.cleanup()

    def enter(self) -> None:
        """
        Called each time the user opens this module.

        This hook can be used to ensure state is up-to-date when the user
        returns to the module. Currently no additional logic is required.
        """
        pass

    def exit(self) -> None:  # noqa: A003
        """
        Called each time the user navigates away from this module.

        This hook can be used to pause or finalize ongoing tasks, but
        no special handling is required at the moment.
        """
        pass
