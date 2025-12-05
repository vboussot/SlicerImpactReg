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
    """
    Thin wrapper around KonfAI's Process class for running Elastix.

    This class parses Elastix stdout/stderr in order to:
      - forward logs to the Slicer UI
      - estimate registration progress from iteration count and timing
    """

    def __init__(self, _update_logs, _update_progress):
        super().__init__(_update_logs, _update_progress)
        # Total number of iterations across all presets (set externally)
        self._total_iterations = 0

    def on_stdout_ready(self):
        """
        Called whenever Elastix writes to stdout.

        We parse lines that start with an iteration index, extract the
        last column as iteration time (ms), and update a global iteration
        counter to compute an overall percentage.
        """
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
        """
        Called whenever download model writes to stderr.

        We forward all messages to the log and update progress if a line
        contains a percentage (e.g. ' 35%').
        """
        line = self.readAllStandardError().data().decode().strip()
        if line:
            line = line.replace("\r\n", "\n").split("\r")[-1]
            self._update_logs(line)
            match = re.search(r"(\d+)%", line)
            if match:
                percent = int(match.group(1))
                self._update_progress(percent, "")

    def set_total_iterations(self, total_iterations: int) -> None:
        """
        Set the total number of Elastix iterations across all presets.

        This is used to convert iteration index into a global progress bar.
        """
        self._total_iterations = total_iterations
        self._it = 0


#
# ImpactReg
#
class ImpactReg(ScriptedLoadableModule):
    """
    Slicer module entry point for IMPACT-Reg.

    This only declares metadata (name, category, contributors, help,
    acknowledgments). The actual UI lives in ImpactRegWidget below.
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
        # Help text (displayed in the module panel)
        self.parent.helpText = _(
            "<p>"
            "Slicer IMPACT-Reg is a 3D Slicer module dedicated to <b>multimodal medical image registration</b>."
            " It integrates the <b>IMPACT</b> deep semantic similarity metric within the Elastix registration engine,"
            " and exposes predefined registration presets through a simple graphical interface."
            "</p>"
            "<p>"
            "With this module you can:<br>"
            "&bull; Run automated IMPACT-based registration pipelines (e.g., CT–CBCT, MR–CT)<br>"
            "&bull; Use GPU-accelerated feature extraction when available<br>"
            "&bull; Evaluate registration quality using landmarks, segmentations, and intensity-based metrics<br>"
            "&bull; Visualize warped images, labels, and deformation fields directly in Slicer<br>"
            "&bull; Estimate registration uncertainty from ensembles of registration presets"
            "</p>"
            "<p>"
            "Registration presets and pretrained feature extractors are distributed as KonfAI Apps and parameter"
            " packages (downloaded automatically from Hugging Face), so that workflows remain "
            "reproducible and easy to update."
            "</p>"
        )

        # Acknowledgment text (displayed in the About section)
        self.parent.acknowledgementText = _(
            "<p>This module was originally developed by Valentin Boussot "
            "(University of Rennes, France).</p>"
            "<p>It integrates the IMPACT similarity metric for multimodal registration and uses the "
            "KonfAI deep learning framework for feature extraction and workflow management.</p>"
            "<p>If you use this module in your research, please cite:<br>"
            "Boussot V. et al.:<br>"
            "<b>IMPACT: A Generic Semantic Loss for Multimodal Medical Image Registration.</b><br>"
            "arXiv:2503.24121, 2025.<br><br>"
            "Boussot V., Dillenseger J.-L.:<br>"
            "<b>KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging.</b><br>"
            '<a href="https://arxiv.org/abs/2508.09823">https://arxiv.org/abs/2508.09823</a>'
            "</p>"
        )


def resource_path(filename):
    """Return the absolute path of the module ``Resources`` directory for ImpactReg."""
    scripted_modules_path = os.path.dirname(slicer.modules.impactreg.path)
    return os.path.join(scripted_modules_path, "Resources", filename)


def resource_konfai_path(filename):
    """Return the absolute path of the module ``Resources`` directory for SlicerKonfAI."""
    scripted_modules_path = os.path.dirname(slicer.modules.konfai.path)
    return os.path.join(scripted_modules_path, "Resources", filename)


class Preset:
    """
    Wrapper around an IMPACT-Reg preset.

    A preset bundles:
      - Elastix parameter maps
      - one or several TorchScript models for IMPACT features
      - a preprocess function (Python code downloaded from HF)
      - metadata (iterations, description)
    """

    def __init__(self, repo_id: str, metadata: dict[str, str]) -> None:
        self._display_name = metadata["display_name"]

        # Download all parameter maps for this preset from HF
        self._parameter_maps = []
        for parameter_map in metadata["parameter_maps"]:
            self._parameter_maps.append(
                hf_hub_download(repo_id=repo_id, filename=parameter_map, repo_type="model", revision=None)
            )  # nosec B615

        # Lazy-install TorchScript models on first use
        self._models_names = metadata["models"]
        self._models: list[str] = []

        # Optional Python preprocess function referenced as "<file>:<function>"
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
            # Identity preprocess if no function is provided
            self._preprocess_function = lambda x: x

        self._iterations = int(metadata["iterations"])
        self._short_description = metadata["short_description"]
        self._description = metadata["description"]

    def get_display_name(self):
        """Human-readable name used in the preset combo box."""
        return self._display_name

    def install(self) -> tuple[list[str], list[str]]:
        """
        Ensure all TorchScript models associated with this preset are present locally.

        Returns
        -------
        parameter_maps : list[str]
            Paths to Elastix parameter map files.
        models : list[str]
            Paths to TorchScript models to be copied next to the working directory.
        """
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
        """Apply the preset's preprocess function to a SimpleITK image."""
        return self._preprocess_function(image)

    def get_models_name(self):
        """Return model filenames (without repo prefix) for linking in the work directory."""
        return [model_name.split(":")[1] for model_name in self._models_names]

    def get_number_of_iterations(self) -> int:
        """Return the number of Elastix iterations configured for this preset."""
        return self._iterations

    def set_device(self, parametermap_path: str, device_str: str | None):
        """
        Update the ImpactGPU field of a parameter map with the selected device index.

        device_str is an optional comma-separated string (e.g. '0,1') coming from the GUI.
        Only the first index is used here.
        """
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
        """Return the long description for this preset."""
        return self._description

    def get_short_description(self) -> str:
        """Return the short description displayed by default in the UI."""
        return self._short_description

    def get_preset_dir(self) -> Path:
        """
        Return the directory where the main parameter map resides.

        This is used to open the preset folder in the file explorer.
        """
        return Path(self._parameter_maps[0])


class ElastixImpactWidget(AppTemplateWidget):
    """
    KonfAI app widget responsible for driving Elastix-based IMPACT registration.

    This class connects:
      - the Slicer widgets defined in ElastixImpactReg.ui
      - the KonfAI processing backend (Process, KonfAIMetricsPanel)
      - Hugging Face presets and models
    """

    def __init__(self, name: str, repo_id: str):
        super().__init__(name, slicer.util.loadUI(resource_path("UI/ElastixImpactReg.ui")))
        self.repo_id = repo_id
        self._elastix_bin: None | Path = None

        # QA panels (with/without reference metrics)
        self.evaluation_panel = KonfAIMetricsPanel()
        self.ui.withRefMetricsPlaceholder.layout().addWidget(self.evaluation_panel)
        self.uncertainty_panel = KonfAIMetricsPanel()
        self.ui.noRefMetricsPlaceholder.layout().addWidget(self.uncertainty_panel)

        self._description_expanded = False

        # Wire all MRML node selectors to the parameter node updater
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

        # Buttons and tabs
        self.ui.toggleDescriptionButton.clicked.connect(self.on_toggle_description)
        self.ui.runRegistrationButton.clicked.connect(self.on_run_registration_button)
        self.ui.runEvaluationButton.clicked.connect(self.on_run_evaluation_button)

        self.ui.qaTabWidget.currentChanged.connect(self.on_tab_changed)

        # Preset configuration button (wheel icon)
        icon_path = resource_konfai_path("Icons/gear.png")
        self.ui.presetButton.setIcon(QIcon(icon_path))
        self.ui.presetButton.setIconSize(QSize(18, 18))
        self.ui.presetButton.clicked.connect(self.on_open_config)

        # Load preset database from Hugging Face
        preset_database_path = hf_hub_download(
            repo_id=repo_id, filename="PresetDatabase.json", repo_type="model", revision=None
        )  # nosec B615

        with open(preset_database_path, encoding="utf-8") as f:
            preset_database = json.load(f)

        # Populate the preset combo with Preset objects as userData
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

    def setup_preset_chips(self, presets: list[str]):
        """
        Setup the list of selected presets as removable 'chips' in the UI.

        This gives a quick visual overview of which presets will be run
        in sequence, and allows users to remove them with a single click.
        """
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
            """
            Insert the chip before the layout's trailing spacer item so
            that chips stay aligned to the left.
            """
            insert_index = layout.count()
            for i in range(layout.count()):
                if layout.itemAt(i).spacerItem() is not None:
                    insert_index = i
                    break
            layout.insertWidget(insert_index, widget)

        def add_chip(text):
            """
            Create and insert a new chip button for the given preset name.
            """
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

        # Recreate chips from parameter node on initialization
        for preset in presets:
            add_chip(preset)

    def get_selected_presets(self):
        """
        Retrieve the list of preset names currently represented as chips.
        """
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
        Open configuration directory for the currently selected preset.

        This allows advanced users to inspect or edit the Elastix parameter
        maps and associated configuration files.
        """
        preset_dir = self.ui.parameterMapPresetComboBox.currentData.get_preset_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(preset_dir.parent))

    def on_preset_selected(self):
        """
        Called when the user selects a different preset in the combo box.

        We collapse the description by default and disable QA tabs until a
        registration has been run with the new preset configuration.
        """
        self.ui.removePresetButton.setEnabled(False)
        self._description_expanded = False
        self.on_toggle_description()
        self.set_parameter("Preset", str(self.ui.parameterMapPresetComboBox.currentIndex))

        self.ui.qaTabWidget.setTabEnabled(1, False)

    def on_toggle_description(self):
        """
        Toggle between short and full description of the current preset.
        """
        preset: Preset = self.ui.parameterMapPresetComboBox.currentData
        if self._description_expanded:
            self.ui.presetDescriptionLabel.setText(preset.get_description())
            self.ui.toggleDescriptionButton.setText("Less ▲")
        else:
            self.ui.presetDescriptionLabel.setText(preset.get_short_description())
            self.ui.toggleDescriptionButton.setText("More ▼")
        self._description_expanded = not self._description_expanded

    def app_setup(self, update_logs, update_progress, parameter_node):
        """
        Initialize the app-level process and parameter node.

        This is called by the KonfAICoreWidget when the app is created.
        """
        self._update_logs = update_logs
        self._update_progress = update_progress
        self._parameter_node = parameter_node
        self.process = ElastixProcess(update_logs, update_progress)

    def initialize_parameter_node(self):
        """
        Initialize the parameter node with default values for this app.

        We auto-select the first two scalar volume nodes in the scene
        as fixed/moving volumes if nothing is set yet, and ensure a
        default preset index is stored.
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
        """
        Called when the user enters the app tab inside SlicerKonfAI.

        We simply re-apply the current preset selection logic.
        """
        super().enter()
        self.on_preset_selected()

    def update_gui_from_parameter_node(self):
        """
        Refresh button states and tooltips based on current parameter node.

        This is called whenever something changes in the scene or parameter node.
        """
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

        # Enable/disable the evaluation button depending on QA mode
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
        """
        Push current GUI state into the parameter node.

        This keeps the module state serializable and allows scene save/load.
        """
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

    def on_run_evaluation_button(self):
        """
        Dispatch evaluation logic depending on the selected QA tab:

          - 'withRefTab': run evaluation with reference data (images, seg, fiducials)
          - other tab: run uncertainty estimation
        """
        self.evaluation_panel.clear_images_list()
        self.uncertainty_panel.clear_images_list()
        self.on_run_button(
            self.evaluation if self.ui.qaTabWidget.currentWidget().name == "withRefTab" else self.uncertainty
        )

    def next_evaluation(self, args_list: list[list[str]]):
        """
        Execute a list of KonfAI evaluation commands sequentially.

        Each entry in args_list is a konfai-apps CLI call. Once one finishes,
        we parse metrics and images, then move to the next.
        """
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
        """
        Build and run evaluation workflows based on the selected reference data.

        Depending on what the user has specified, we can evaluate:
          - warped images (fixed vs warped moving)
          - warped segmentations
          - warped landmarks
        Each case results in a konfai-apps 'eval' call with the appropriate YAML.
        """
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

        # --- Image-based metrics ---
        if (
            fixed_image_evaluation
            and fixed_image_evaluation.GetImageData()
            and moving_image_evaluation
            and moving_image_evaluation.GetImageData()
        ):
            # Export fixed image
            volume_storage_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
            volume_storage_node.SetFileName(str(self._work_dir / "FixedImage.mha"))
            volume_storage_node.UseCompressionOff()
            volume_storage_node.WriteData(fixed_image_evaluation)
            volume_storage_node.UnRegister(None)

            # Create warped moving image using the selected transform
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
        # --- Segmentation-based metrics ---
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
        # --- Landmark-based metrics ---
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

            # Warp moving landmarks using the Slicer transform logic
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

        # Optional mask used in all evaluations
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
        """
        Placeholder for future uncertainty estimation based on transform ensembles.

        The idea is to:
          - combine all transforms from the sequence node
          - feed them to a KonfAI uncertainty workflow

        Not implemented yet; kept as a stub for future development.
        """

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

    def on_run_registration_button(self):
        """Entry point for the 'Run' button: start or stop registration."""
        self.on_run_button(self.registration)

    def get_elastix_bin(self) -> None:
        """
        Locate or download the Elastix binary bundled with the extension.

        If the binary is found locally, we directly start registration.
        Otherwise we run the Download.py helper using PythonSlicer, then
        retry registration once the download is complete.
        """
        file = "elastix-impact-{}-shared-with-deps-{}".format(
            "win64" if platform.system() == "Windows" else "linux", "cu126" if torch.cuda.is_available() else "cpu"
        )
        path = Path(os.path.dirname(os.path.abspath(__file__))) / "Resources" / "bin"
        executable = "elastix.exe" if platform.system() == "Windows" else "elastix"
        matches = list((path / file).rglob(executable))
        if len(matches):
            self._elastix_bin = matches[0]
            file_path = Path(self._elastix_bin)
            # Ensure execute permission on Unix-like systems
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
            matches = list((path / file).rglob(executable))
            if len(matches):
                self._elastix_bin = matches[0]
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
        """
        Run Elastix sequentially for a list of presets.

        Each preset:
          - preprocesses fixed/moving images
          - installs required TorchScript models
          - appends its parameter maps to the Elastix command

        The resulting transform is stored and used to build a sequence for
        uncertainty / composite-transform analysis.
        """
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

        # Clean working directory (except MHA inputs)
        for f in self._work_dir.iterdir():
            if f.suffix != ".mha":
                if f.is_file():
                    f.unlink()
                else:
                    shutil.rmtree(f)
        # Copy models next to the work directory using the filenames expected by IMPACT
        for model_path, model_name in zip(models_path, preset.get_models_name()):
            link_path = self._work_dir / model_name
            if not link_path.exists():
                link_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(model_path, link_path)

        # Copy parameter maps and set GPU device
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

                # Expose each transform as a Slicer node and add it to a sequence
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
        """
        Called once all presets have been executed.

        We build:
          - a SimpleITK displacement field for each transform
          - their average transform, saved as a new Slicer transform
          - a warped moving image volume using this average transform
        """
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

        # Copy the average transform into the user-selected output transform node
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
        """
        Top-level registration entry point.

        If Elastix is not installed yet, trigger the download, otherwise
        build the Elastix command-line arguments and execute presets in sequence.
        """
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

        # Collect selected presets in the order displayed in the combo box
        selected_presets = self.get_selected_presets()
        combo = self.ui.parameterMapPresetComboBox
        presets = []
        for i in range(combo.count):
            if combo.itemText(i) in selected_presets:
                presets.append(combo.itemData(i))

        # Compute total number of iterations for progress monitoring
        total_it = 0
        for preset in presets:
            total_it += preset.get_number_of_iterations()
        self.process.set_total_iterations(total_it)

        # Clear previous transform sequence
        sequence_node = self.ui.inputTransformSequenceSelector.currentNode()
        if sequence_node is not None:
            sequence_node.RemoveAllDataNodes()

        # Start chained registration
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
