# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813
import sitkUtils
import slicer
from KonfAI import (
    AppTemplateWidget,
    ChipSelector,
    KonfAICoreWidget,
    KonfAIMetricsPanel,
    Process,
    RemoteServer,
    _is_reload_setup,
    has_node_content,
)
from qt import QDesktopServices, QIcon, QSize, QUrl, QWidget
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget


class ElastixProcess(Process):
    """
    Thin wrapper around KonfAI's Process class for running Elastix.

    This class parses Elastix stdout/stderr in order to:
      - forward logs to the Slicer UI
      - estimate registration progress from iteration count and timing
    """

    def __init__(
        self,
        update_logs: Callable[[str, bool], None],
        update_progress: Callable[[int, float], None],
        running_setter: Callable[[bool], None],
    ):
        super().__init__(update_logs, update_progress, running_setter)
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

            m = re.search(r"(\d{1,3})%", line)
            if m:
                pct = int(m.group(1))
                self._update_progress(pct, "")
                return

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
            match = re.search(r"(\d+)%\|", line)
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
            '<a href="https://arxiv.org/abs/2503.24121">https://arxiv.org/abs/2503.24121</a><br>'
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
      - metadata (iterations, description)
    """

    def __init__(self, repo_id: str, metadata: dict[str, str], force_update: bool = False) -> None:
        self._display_name = metadata["display_name"]

        # Download all parameter maps for this preset from HF
        self._parameter_maps: list[Path] = []

        for parameter_map in metadata["parameter_maps"]:
            self._parameter_maps.append(self._download(repo_id, parameter_map, force_update))

        # Lazy-install TorchScript models on first use
        self._models_names = metadata["models"]

        self._iterations = int(metadata["iterations"])
        self._short_description = metadata["short_description"]
        self._description = metadata["description"]

    def get_parameter_maps(self) -> list[Path]:
        return self._parameter_maps

    def _download(self, repo_id: str, filename: str, force_update: bool) -> Path:
        from huggingface_hub import hf_hub_download

        if force_update:
            try:
                return hf_hub_download(
                    repo_id=repo_id, filename=filename, repo_type="model", revision=None
                )  # nosec B615
            except Exception as e1:
                raise NameError(
                    f"Unable to load parameter map '{filename}' from Hugging Face "
                    f"repository '{repo_id}'. The file was not found in the local cache and "
                    "could not be downloaded from the Hub.\n"
                    "Please check that the repository exists, that it contains this file, "
                    "and that you have network access.\n"
                    f"Original error: {e1}"
                )
        else:
            try:
                return hf_hub_download(
                    repo_id=repo_id, filename=filename, repo_type="model", revision=None, local_files_only=True
                )  # nosec B615
            except Exception:
                return self._download(repo_id, filename, True)

    def get_display_name(self):
        """Human-readable name used in the preset combo box."""
        return self._display_name

    def get_models_name(self):
        """Return model filenames (without repo prefix) for linking in the work directory."""
        return self._models_names

    def get_number_of_iterations(self) -> int:
        """Return the number of Elastix iterations configured for this preset."""
        return self._iterations

    def set_device(self, parametermap_path: str, device_str: list[str]):
        """
        Update the ImpactGPU field of a parameter map with the selected device index.

        device_str is an optional comma-separated string (e.g. '0,1') coming from the GUI.
        Only the first index is used here.
        """
        device = -1
        if device_str:
            device = int(device_str[0])

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
        self._elastix_bin = (
            Path(resource_path("bin"))
            / "elastix-impact"
            / ("elastix.exe" if platform.system() == "Windows" else (Path("bin") / "elastix"))
        )

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
        self.ui.referenceVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.update_parameter_node_from_gui)

        # Buttons and tabs
        self.ui.toggleDescriptionButton.clicked.connect(self.on_toggle_description)
        self.ui.runRegistrationButton.clicked.connect(self.on_run_registration_button)
        self.ui.runEvaluationButton.clicked.connect(self.on_run_evaluation_button)

        self.ui.qaTabWidget.currentChanged.connect(self.on_tab_changed)

        # Preset configuration button (wheel icon)
        self.ui.presetButton.setIcon(QIcon(resource_konfai_path("Icons/gear.png")))
        self.ui.presetButton.setIconSize(QSize(18, 18))
        self.ui.presetButton.clicked.connect(self.on_open_config)

        self.ui.refreshPresetsListButton.setIcon(QIcon(resource_konfai_path("Icons/refresh.png")))
        self.ui.refreshPresetsListButton.setIconSize(QSize(18, 18))
        self.ui.refreshPresetsListButton.clicked.connect(self.on_refresh_presets)

        self.ui.addPresetButton.setEnabled(False)
        self.ui.removePresetButton.setEnabled(False)

        self.ui.removePresetButton.clicked.connect(self.on_remove_preset)
        self.ui.addPresetButton.clicked.connect(self.on_add_preset)

        self.chip_selector = ChipSelector(
            self.ui.parameterMapPresetComboBox,
            self.ui.selectedPresetsWidget.layout(),
            combo_remove=False,
            on_change=self.on_preset_selected_change,
        )
        self.presets: dict[str, Preset] = {}
        self._current_preset = None

    def on_remove_preset(self) -> None:
        pass

    def on_add_preset(self) -> None:
        pass

    def on_refresh_presets(self) -> None:
        self.populate_presets(True)

    def populate_presets(self, force_update: bool = False) -> None:
        from huggingface_hub import hf_hub_download

        if force_update:
            try:
                # Load preset database from Hugging Face
                preset_database_path = hf_hub_download(
                    repo_id=self.repo_id, filename="PresetDatabase.json", repo_type="model", revision=None
                )  # nosec B615
            except Exception as e1:
                slicer.util.errorDisplay(
                    f"Unable to load 'PresetDatabase.json' from Hugging Face repository "
                    f"'{self.repo_id}'. The file was not found in the local cache and could "
                    "not be downloaded from the Hub.\n"
                    "Please check that the repository exists, that it contains a "
                    "'PresetDatabase.json' file, and that you have network access.\n",
                    detailedText=getattr(e1, "details", None) or str(e1),
                )
                return
        else:
            try:
                preset_database_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename="PresetDatabase.json",
                    repo_type="model",
                    revision=None,
                    local_files_only=True,
                )  # nosec B615
            except Exception:
                self.populate_presets(True)
                return

        with open(preset_database_path, encoding="utf-8") as f:
            preset_database = json.load(f)

        # Populate the preset combo with Preset objects as userData
        self.ui.parameterMapPresetComboBox.clear()
        for preset_metadata in preset_database["presets"]:
            preset = Preset(self.repo_id, preset_metadata, force_update)
            self.presets[preset.get_display_name()] = preset
            self.ui.parameterMapPresetComboBox.addItem(preset.get_display_name())

    def on_remote_server_changed(self) -> None:
        pass

    def on_preset_selected_change(self, preset_selected: list[str]):
        self.update_gui_from_parameter_node()

    def on_tab_changed(self) -> None:
        """
        Update GUI state when the user switches between QA tabs.

        Ensures that button enabling/disabling is consistent with the current tab.
        """
        self.update_gui_from_parameter_node()

    def on_open_config(self):
        """
        Open configuration directory for the currently selected preset.

        This allows advanced users to inspect or edit the Elastix parameter
        maps and associated configuration files.
        """

        preset_dir = self.presets[self.ui.parameterMapPresetComboBox.currentText].get_preset_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(preset_dir.parent))

    def on_preset_selected(self):
        """
        Called when the user selects a different preset in the combo box.

        We collapse the description by default and disable QA tabs until a
        registration has been run with the new preset configuration.
        """
        self._description_expanded = False
        self.on_toggle_description()

    def on_toggle_description(self):
        """
        Toggle between short and full description of the current preset.
        """
        preset = self.presets[self.ui.parameterMapPresetComboBox.currentText]

        if self._description_expanded:
            self.ui.presetDescriptionLabel.setText(preset.get_description())
            self.ui.toggleDescriptionButton.setText("Less ▲")
        else:
            self.ui.presetDescriptionLabel.setText(preset.get_short_description())
            self.ui.toggleDescriptionButton.setText("More ▼")
        self._description_expanded = not self._description_expanded

    def app_setup(
        self,
        update_logs,
        update_progress,
        parameter_node,
        begin_status_progress: Callable[[], None] | None = None,
        end_status_progress: Callable[[], None] | None = None,
    ) -> None:
        """
        Initialize the app-level process and parameter node.

        This is called by the KonfAICoreWidget when the app is created.
        """
        self._update_logs = update_logs
        self._update_progress = update_progress
        self._parameter_node = parameter_node
        self.process = ElastixProcess(update_logs, update_progress, self.set_running)

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

        self.ui.referenceVolumeSelector.setCurrentNode(self.get_parameter_node("ReferenceVolume"))

    def enter(self):
        """
        Called when the user enters the app tab inside SlicerKonfAI.

        We simply re-apply the current preset selection logic.
        """

        if self.ui.parameterMapPresetComboBox.count == 0:
            self.populate_presets()
            self.ui.parameterMapPresetComboBox.currentIndexChanged.connect(self.on_preset_selected)

        super().enter()
        self.on_preset_selected()

    def update_gui_from_parameter_node(self):
        """
        Refresh button states and tooltips based on current parameter node.

        This is called whenever something changes in the scene or parameter node.
        """
        fixed_volume = self.get_parameter_node("FixedVolume")
        moving_volume = self.get_parameter_node("MovingVolume")
        if has_node_content(fixed_volume) and has_node_content(moving_volume) and self.chip_selector.selected():
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
        reference_volume = self.get_parameter_node("ReferenceVolume")
        # Enable/disable the evaluation button depending on QA mode
        if self.ui.qaTabWidget.currentWidget().name == "withRefTab":
            if (
                (has_node_content(fixed_image_evaluation) and has_node_content(moving_image_evaluation))
                or (has_node_content(fixed_seg_evaluation) and has_node_content(moving_seg_evaluation))
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

            if (
                transform_sequence
                and transform_sequence.GetNumberOfDataNodes() > 1
                and has_node_content(reference_volume)
            ):
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
        self.set_parameter_node("ReferenceVolume", self.ui.referenceVolumeSelector.currentNodeID)
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
            args += ["--gpu"] + self.get_device()
        else:
            args += ["--cpu", "1"]

        def on_end_evaluation() -> None:
            try:
                from konfai.evaluator import Statistics

                statistics = Statistics((self._work_dir / "Evaluation").rglob("*.json").__next__())
                self.evaluation_panel.set_metrics(statistics.read())
                self.evaluation_panel.refresh_images_list(
                    Path((self._work_dir / "Evaluation").rglob("*.mha").__next__().parent)
                )
                if len(args_list) > 0:
                    self.next_evaluation(args_list)
            except Exception as e:
                print(e)

        self.process.run("konfai-apps", self._work_dir, args, on_end_evaluation)

    def evaluation(self, remote_server: RemoteServer | None, devices: list[str]):
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
        if has_node_content(fixed_image_evaluation) and has_node_content(moving_image_evaluation):
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
        if has_node_content(fixed_seg_evaluation) and has_node_content(moving_seg_evaluation):

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
        if has_node_content(mask_evaluation):
            volume_storage_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
            volume_storage_node.SetFileName(str(self._work_dir / "Mask.mha"))
            volume_storage_node.UseCompressionOff()
            volume_storage_node.WriteData(mask_evaluation)
            volume_storage_node.UnRegister(None)

            for args in args_list:
                args += ["--mask", "Mask.mha"]
        if remote_server is not None:
            args += ["--host", remote_server.host, "--port", remote_server.port, "--token", remote_server.token]

        self.next_evaluation(args_list)

    def uncertainty(self, remote_server: RemoteServer | None, devices: list[str]):
        """
        Placeholder for future uncertainty estimation based on transform ensembles.

        The idea is to:
          - combine all transforms from the sequence node
          - feed them to a KonfAI uncertainty workflow

        Not implemented yet; kept as a stub for future development.
        """

        self.uncertainty_panel.clear_metrics()
        transform_sequence_node = self.ui.inputTransformSequenceSelector.currentNode()

        transform_path_tmp = self._work_dir / "Transforms"
        transform_path_tmp.mkdir(parents=True, exist_ok=True)
        transform_to_displacementfield_filter = sitk.TransformToDisplacementFieldFilter()
        images = []
        reference_image = sitkUtils.PullVolumeFromSlicer(self.ui.referenceVolumeSelector.currentNode())
        for i in range(transform_sequence_node.GetNumberOfDataNodes()):
            tnode = transform_sequence_node.GetNthDataNode(i)
            slicer.util.saveNode(tnode, str(self._work_dir / f"t_{i:05d}.h5"))
            t = sitk.ReadTransform(str(self._work_dir / f"t_{i:05d}.h5"))
            os.remove(str(self._work_dir / f"t_{i:05d}.h5"))
            transform_to_displacementfield_filter.SetReferenceImage(reference_image)
            images.append(transform_to_displacementfield_filter.Execute(t))

        arrays = [sitk.GetArrayFromImage(img) for img in images]
        stack = np.stack(arrays, axis=-1)
        image = sitk.GetImageFromArray(stack, isVector=True)
        sitk.WriteImage(image, str(self._work_dir / "DVFs.mha"))

        args = [
            "uncertainty",
            f"{self.repo_id}:ImpactReg",
            "-i",
            "DVFs.mha",
            "-o",
            "Uncertainty",
        ]

        if self.get_device():
            args += ["--gpu"] + self.get_device()
        else:
            args += ["--cpu", "1"]

        if remote_server is not None:
            args += ["--host", remote_server.host, "--port", remote_server.port, "--token", remote_server.token]

        def on_end_function() -> None:
            from konfai.evaluator import Statistics

            statistics = Statistics((self._work_dir / "Uncertainty").rglob("*.json").__next__())
            self.uncertainty_panel.set_metrics(statistics.read())
            self.uncertainty_panel.refresh_images_list(
                Path((self._work_dir / "Uncertainty").rglob("*.mha").__next__().parent)
            )

        self.process.run("konfai-apps", self._work_dir, args, on_end_function)

    def on_run_registration_button(self):
        """Entry point for the 'Run' button: start or stop registration."""
        self.on_run_button(self.registration)

    def try_elastix(self) -> str:
        try:
            subprocess.run(
                [str(self._elastix_bin), "-h"],
                capture_output=True,
                text=True,
                check=True,
            )
            return ""

        except subprocess.CalledProcessError as e:
            msg = "Elastix execution failed.\n\n"

            msg += f"Command:\n{' '.join(e.cmd)}\n"
            msg += f"Return code: {e.returncode}\n\n"

            if e.stderr:
                msg += "Error output:\n"
                msg += e.stderr.strip()
            return msg

        except OSError as e:
            msg = (
                "Elastix could not be started.\n\n"
                "This is usually caused by missing shared libraries "
                "(e.g. LibTorch or CUDA runtime).\n\n"
                f"System error:\n{str(e)}"
            )

            return msg

    def install_elastix_bin(self, remote_server: RemoteServer | None, devices: list[str]) -> None:
        """
        Locate or download the Elastix binary bundled with the extension.

        If the binary is found locally, we directly start registration.
        Otherwise we run the Download.py helper using PythonSlicer, then
        retry registration once the download is complete.
        """

        def on_en_function():
            if not self._elastix_bin.exists():
                raise FileNotFoundError("Elastix binary not found. Installation failed.")

            self.registration(remote_server, devices)

        path = Path(resource_path("bin"))

        if (path / "elastix-impact").exists():
            shutil.rmtree(path / "elastix-impact")
        self.process.run(shutil.which("PythonSlicer"), path, ["install.py"], on_en_function)

    def next_registration(
        self,
        presets: list[Preset],
        args_init: list[str],
        fixed_image_node,
        moving_image_node,
        transforms: list[sitk.Transform],
    ) -> None:
        """
        Run Elastix sequentially for a list of presets.

        Each preset:
          - installs required TorchScript models
          - appends its parameter maps to the Elastix command

        The resulting transform is stored and used to build a sequence for
        uncertainty / composite-transform analysis.
        """
        preset = presets.pop(0)
        args = args_init.copy()

        parameter_maps_path = preset.get_parameter_maps()

        models_path = []
        from huggingface_hub import hf_hub_download

        for model_name in preset.get_models_name():
            try:
                if ":" in model_name:
                    model_path = hf_hub_download(
                        repo_id=model_name.split(":")[0],
                        filename=model_name.split(":")[1],
                        repo_type="model",
                        revision=None,
                        local_files_only=True,
                    )  # nosec B615
                else:
                    model_path = Path(model_name)

                models_path.append(model_path)
            except Exception:
                try:
                    models_path.append(
                        hf_hub_download(
                            repo_id=model_name.split(":")[0],
                            filename=model_name.split(":")[1],
                            repo_type="model",
                            revision=None,
                        )
                    )  # nosec B615
                except Exception as e:
                    slicer.util.errorDisplay(
                        f"Unable to load '{model_name.split(":")[1]}' from Hugging Face repository "
                        f"'{model_name.split(":")[0]}'. The file was not found in the local cache and could "
                        "not be downloaded from the Hub.\n"
                        "Please check that the repository exists and that you have network access.\n",
                        detailedText=getattr(e, "details", None) or str(e),
                    )

        sitk.WriteImage(sitkUtils.PullVolumeFromSlicer(fixed_image_node), str(self._work_dir / "FixedImage.mha"))
        sitk.WriteImage(
            sitkUtils.PullVolumeFromSlicer(moving_image_node),
            str(self._work_dir / "MovingImage.mha"),
        )

        # Clean working directory (except MHA inputs)
        for f in self._work_dir.iterdir():
            if f.suffix != ".mha":
                if f.is_file():
                    f.unlink()
                else:
                    shutil.rmtree(f)
        # Copy models next to the work directory using the filenames expected by IMPACT
        for model_path, model_name in zip(
            models_path, [name.split(":")[1] if ":" in name else Path(name).name for name in preset.get_models_name()]
        ):
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

    def registration(self, remote_server: RemoteServer | None, devices: list[str]) -> None:
        """
        Top-level registration entry point.

        If Elastix is not installed yet, trigger the download, otherwise
        build the Elastix command-line arguments and execute presets in sequence.
        """

        if not self._elastix_bin.exists():
            reply = slicer.util.confirmYesNoDisplay(
                "IMPACT Reg requires Elastix with the IMPACT module.\n\n"
                "Elastix is not currently installed on your system.\n\n"
                "Would you like to install it now?",
                windowTitle="IMPACT – Elastix Installation",
            )

            if reply:
                self.install_elastix_bin(remote_server, devices)
            return

        msg = self.try_elastix()

        if msg:
            reply = slicer.util.confirmYesNoDisplay(
                msg + "\n\nDo you want to reinstall Elastix now?", windowTitle="IMPACT / Elastix error"
            )

            if reply:
                if not (Path(resource_path("bin")) / "elastix-impact").exists():
                    shutil.rmtree(Path(resource_path("bin")) / "elastix-impact")
                self.install_elastix_bin(remote_server, devices)
                return
            else:
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

        if has_node_content(fixed_mask_node):
            sitk.WriteImage(sitkUtils.PullVolumeFromSlicer(fixed_mask_node), str(self._work_dir / "FixedMask.mha"))
            args_init += ["-fMask", "FixedMask.mha"]

        if has_node_content(moving_mask_node):
            sitk.WriteImage(sitkUtils.PullVolumeFromSlicer(moving_mask_node), str(self._work_dir / "MovingMask.mha"))
            args_init += ["-mMask", "MovingMask.mha"]

        # Collect selected presets in the order displayed in the combo box
        selected_presets = self.chip_selector.selected()
        presets = []
        for preset_name in selected_presets:
            presets.append(self.presets[preset_name])

        # Compute total number of iterations for progress monitoring
        total_it = 0
        for preset in presets:
            total_it += preset.get_number_of_iterations()
        self.process.set_total_iterations(total_it)

        # Clear previous transform sequence
        sequence_node = self.ui.inputTransformSequenceSelector.currentNode()
        if sequence_node is not None:
            sequence_node.RemoveAllDataNodes()
        transforms: list[sitk.Transform] = []
        # Start chained registration
        self.next_registration(
            presets,
            args_init.copy(),
            self.ui.fixedVolumeSelector.currentNode(),
            self.ui.movingVolumeSelector.currentNode(),
            transforms,
        )


class ImpactRegWidget(ScriptedLoadableModuleWidget):
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

        if _is_reload_setup("SlicerImpactReg"):
            self.konfai_core.enter()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.konfai_core.cleanup()

    def enter(self) -> None:
        """
        Called each time the user opens this module.

        This hook can be used to ensure state is up-to-date when the user
        returns to the module. Currently no additional logic is required.
        """
        self.konfai_core.enter()

    def exit(self) -> None:  # noqa: A003
        """
        Called each time the user navigates away from this module.

        This hook can be used to pause or finalize ongoing tasks, but
        no special handling is required at the moment.
        """
        self.konfai_core.exit()
