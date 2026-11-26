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

"""Slicer IMPACT-Reg: multimodal registration through the `impact-reg-konfai` orchestrator.

The module reuses the generic SlicerKonfAI app widget (selection / inference / QA panels)
and only substitutes the two panels that drive a CLI: registration runs
``impact-reg-konfai register`` (the checkpoint chips select the *presets*, ``APP#PRESET``; several presets
are ensembled — their displacement fields are averaged by the CLI), evaluation runs
``impact-reg-konfai eval`` (the CLI warps the moving data with the transform), and
uncertainty runs ``impact-reg-konfai uncertainty`` on the ensemble's stack the registration kept.
"""

import shutil
from pathlib import Path

import SimpleITK as sitk  # noqa: N813
import sitkUtils
import slicer
from KonfAI import (
    KonfAIAppInferencePanel,
    KonfAIAppQAPanel,
    KonfAIAppTemplateWidget,
    KonfAICoreWidget,
    RemoteServer,
    _is_reload_setup,
    install_package,
)
from qt import QComboBox, QHBoxLayout, QLabel, QSpinBox, QWidget
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget

IMPACT_REG_REPO = "VBoussot/ImpactReg"

# The flag of `impact-reg-konfai eval` each evaluation input of the presets' app.json goes to (a Mask* entry: --mask).
EVALUATION_FLAGS = {
    "Transform": "--transform",
    "FixedImage": "-f",
    "MovingImage": "-m",
    "FixedSeg": "--gt-fixed-seg",
    "MovingSeg": "--gt-moving-seg",
    "FixedFid": "--gt-fixed-fid",
    "MovingFid": "--gt-moving-fid",
}

# How `impact-reg-konfai register --out-of-core` registers a pair; auto is its default.
ROUTES = ("auto", "whole", "tiled", "sampled")


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
        self.parent.helpText = _(
            "<p>"
            "Slicer IMPACT-Reg is a 3D Slicer module dedicated to <b>multimodal medical image registration</b>."
            " It drives the <b>IMPACT</b> deep semantic similarity metric across multiple registration engines"
            " (Elastix, ConvexAdam, FireANTs),"
            " and exposes predefined registration presets through a simple graphical interface."
            "</p>"
            "<p>"
            "With this module you can:<br>"
            "&bull; Run automated IMPACT-based registration pipelines (e.g., CT–CBCT, MR–CT)<br>"
            "&bull; Ensemble several registration presets (their displacement fields are averaged)<br>"
            "&bull; Evaluate registration quality using landmarks, segmentations, and intensity-based metrics<br>"
            "&bull; Visualize warped images, labels, and deformation fields directly in Slicer<br>"
            "&bull; Estimate registration uncertainty from ensembles of registration presets"
            "</p>"
            "<p>"
            "Registration presets are distributed as KonfAI Apps (downloaded automatically from Hugging Face),"
            " so that workflows remain reproducible and easy to update."
            "</p>"
        )
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


class RegistrationInferencePanel(KonfAIAppInferencePanel):
    """Inference panel variant that drives ``impact-reg-konfai register``.

    The checkpoint chip selector is repurposed to select registration *presets*
    (``APP#PRESET``, the presets of the repository's apps): selecting several presets runs them all and the
    CLI averages their displacement fields into a single transform.
    """

    def __init__(self, template) -> None:
        super().__init__(template)
        # This panel ensembles *presets* (the repository apps), not model checkpoints. Presets may run
        # different engines (Elastix, ConvexAdam, FireANTs). The app selector at the top picks the main preset (it
        # seeds the ensemble and drives evaluation); the "Ensemble with" combo adds further presets to
        # average in. Hide the inference-only controls that do not apply: the ensemble count and MC-dropout.
        self.ui.label_selected_checkpoints.setText(_("Presets"))
        self.ui.label_ensemble.setText(_("Ensemble with"))
        self.ui.uncertaintyCheckBox.toolTip = _(
            "Keep every preset's displacement field as the ensemble's stack so registration uncertainty (the "
            "ensemble spread) can be estimated in the QA tab: two or more presets, or TTA on a preset that allows it."
        )
        for widget in (self.ui.ensembleSpinBox, self.ui.label_mcdo, self.ui.mcDropoutSpinBox):
            widget.setVisible(False)
        self._add_large_image_row()

    def _add_large_image_row(self) -> None:
        """The row of ``register``'s options for a pair too large for the device, below the sampling row."""
        self._route = QComboBox()
        self._route.addItems(list(ROUTES))
        self._route.toolTip = _(
            "auto: whole where the pair fits the device, sampled where it does not. whole: in one piece on its "
            "native grid, refused if it does not fit. tiled: resampled to fit, then the deformable stage on native "
            "tiles (presets that declare a tile pass). sampled: never resampled nor held whole: the images are read "
            "by boxes."
        )
        self._samples = QSpinBox()
        self._samples.setRange(0, 2**31 - 1)
        self._samples.setSingleStep(65536)
        self._samples.setSpecialValueText(_("engine's"))
        self._samples.toolTip = _("Samples each iteration of the sampled mode draws anew (engine's: its own default).")
        self._region = QSpinBox()
        self._region.setRange(8, 1024)
        self._region.setSingleStep(8)
        self._region.setValue(64)
        self._region.setSuffix(" vx")
        self._region.toolTip = _("Side of a box of the sampled mode, in voxels.")
        self._visits = QSpinBox()
        self._visits.setRange(0, 100000)
        self._visits.setSpecialValueText(_("stage's"))
        self._visits.toolTip = _("Visits of every box in the sampled mode (stage's: its own iterations).")
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for label, widget in (
            (None, self._route),
            (_("Samples"), self._samples),
            (_("Box"), self._region),
            (_("Visits"), self._visits),
        ):
            if label is not None:
                layout.addWidget(QLabel(label))
            layout.addWidget(widget)
        self._route.connect("currentTextChanged(QString)", self._on_route_changed)
        form = self.ui.label_stochastic.parentWidget().layout()
        form.insertRow(5, QLabel(_("Large images")), row)  # under the sampling row (KonfAIAppInference.ui)

    def _on_route_changed(self, route: str) -> None:
        """The sampled mode's options apply where a pair may register sampled."""
        for widget in (self._samples, self._region, self._visits):
            widget.setEnabled(route in ("auto", "sampled"))

    def _large_image_args(self) -> list[str]:
        """``register``'s large-image options as the row sets them."""
        route = self._route.currentText
        args = ["--out-of-core", route]
        if route in ("auto", "sampled"):
            args += ["--region-size", str(self._region.value)]
            args += ["--samples", str(self._samples.value)] if self._samples.value else []
            args += ["--epochs", str(self._visits.value)] if self._visits.value else []
        return args

    def on_app_changed(self, app) -> None:
        super().on_app_changed(app)
        # The preset selector reuses the ensemble widgets, which the base panel hides for a
        # checkpoint-less app — re-show them (the count spin stays hidden: presets are picked, not counted).
        for widget in (
            self.ui.label_ensemble,
            self.ui.checkpointsComboBox,
            self.ui.label_selected_checkpoints,
            self.ui.selectedCheckpointsWidget,
        ):
            widget.setVisible(True)
        self.ui.label_stochastic.setVisible(True)
        # The app selector picks the main preset: it seeds the ensemble (the "Presets" chips) and drives
        # evaluation. Add more presets with the "Ensemble with" combo below; switching the main reseeds
        # the ensemble to it. The CLI averages the displacement fields of all the selected presets.
        presets = self.template.get_preset_names()
        engine = app.get_name().split(":")[-1]
        current = [preset for preset in presets if preset.split("#")[0] == engine][:1]
        self.chip_selector.update(presets, presets, current or presets[:1])

    def _extra_node(self, key: str):
        """Return the node of the extra input selector declared under ``key`` in app.json (or None)."""
        entry = self._extra_input_selectors.get(key)
        return entry[1].currentNode() if entry else None

    def inference(self, remote_server: RemoteServer | None, devices: list[str]) -> None:
        """Run the registration: one ``impact-reg-konfai register`` call with the selected presets. With a server,
        each preset runs there and the moved image and the ensemble are computed here."""
        self.template.evaluation_panel.clear_images_list()
        self.template.uncertainty_panel.clear_images_list()
        self.template.evaluation_panel.clear_metrics()
        self.template.uncertainty_panel.clear_metrics()

        fixed_node = self.ui.inputVolumeSelector.currentNode()
        moving_node = self._extra_node("Moving")
        presets = self.chip_selector.selected()
        if fixed_node is None or moving_node is None or not presets:
            self._update_logs("[ImpactReg] Select a fixed image, a moving image and at least one preset.", False)
            self.template.set_running(False)
            return

        sitk.WriteImage(sitkUtils.PullVolumeFromSlicer(fixed_node), str(self._work_dir / "FixedImage.mha"))
        sitk.WriteImage(sitkUtils.PullVolumeFromSlicer(moving_node), str(self._work_dir / "MovingImage.mha"))
        self._update_logs(f"Inputs saved to temporary folder: {self._work_dir}")

        args = ["register", *presets, "-f", "FixedImage.mha", "-m", "MovingImage.mha"]
        # Optional masks restrict the metric region; the CLI auto-fills a whole-image mask when omitted.
        for flag, key, name in (
            ("--fixed-mask", "FixedMask", "FixedMask.mha"),
            ("--moving-mask", "MovingMask", "MovingMask.mha"),
        ):
            mask_node = self._extra_node(key)
            if mask_node is not None:
                sitk.WriteImage(sitkUtils.PullVolumeFromSlicer(mask_node), str(self._work_dir / name))
                args += [flag, name]
        args += ["-o", "Output"]
        # TTA: each preset app averages flipped registrations internally (standard konfai mechanism).
        if self.ui.ttaSpinBox.value:
            args += ["--tta", str(self.ui.ttaSpinBox.value)]
        # Tuned preset parameters from the Advanced dialog, forwarded to each preset (--set path=value).
        args += self._param_override_set_args()
        # The Uncertainty checkbox gates whether the ensemble's (large) stack of fields is kept, so the QA panel can
        # measure the ensemble spread; without it only the averaged transform is produced.
        if self.ui.uncertaintyCheckBox.isChecked():
            args += ["--uncertainty"]
        args += self._large_image_args()
        if devices:
            args += ["--gpu"] + devices
        else:
            args += ["--cpu", "1"]
        if remote_server is not None:
            args += ["--host", remote_server.host, "--port", str(remote_server.port)]
            if remote_server.token:
                args += ["--token", remote_server.token]

        def on_end_function() -> None:
            case_dir = self._work_dir / "Output" / "P000"
            if not (case_dir / "Transform.h5").exists():
                self._update_logs(
                    f"[ImpactReg] Registration finished but produced no transform under {case_dir}. "
                    "The process probably failed: check the log above for errors.",
                    False,
                )
                return

            # Transform: load it and select it in the QA panel so evaluation uses it directly.
            transform_node = slicer.util.loadTransform(str(case_dir / "Transform.h5"))
            transform_node.SetName(f"{moving_node.GetName()}_to_{fixed_node.GetName()}")
            self.template.ui.inputTransformSelector.setCurrentNode(transform_node)

            # Moved image: load and overlay it on the fixed image.
            moved_node = sitkUtils.PushVolumeToSlicer(
                sitk.ReadImage(str(case_dir / "Moved.mha")), name=f"{moving_node.GetName()}_moved"
            )
            self.ui.outputVolumeSelector.setCurrentNode(moved_node)
            slicer.util.setSliceViewerLayers(
                background=fixed_node, foreground=moved_node, foregroundOpacity=0.5, fit=True
            )

            # The ensemble's stack, kept under Uncertainty, for the QA panel's uncertainty.
            self.template.keep_stack(next(case_dir.glob("InferenceStack*"), None))

            self._update_logs("Registration results loaded into Slicer.")

        self.process.run("impact-reg-konfai", self._work_dir, args, on_end_function)


class RegistrationQAPanel(KonfAIAppQAPanel):
    """QA panel variant that drives ``impact-reg-konfai eval`` and ``impact-reg-konfai uncertainty``.

    Unlike the generic panel, the raw (unwarped) nodes are exported: the CLI applies
    the transform to the moving data itself (images/segmentations/landmarks).
    """

    def __init__(self, template) -> None:
        super().__init__(template)
        # Uncertainty reads the stack the registration kept (``keep_stack``), not a sequence picked here.
        for widget in (self.ui.label_stackVolume, self.ui.inputVolumeSequenceSelector):
            widget.setVisible(False)

    def _export_node(self, node, base: str) -> str | None:
        """Export a node for the CLI: markups as .fcsv, segmentations as labelmap .mha, transforms as .h5, volumes
        as .mha."""
        if node is None:
            return None
        if node.IsA("vtkMRMLTransformNode"):
            file_name = f"{base}.h5"
            slicer.util.saveNode(node, str(self._work_dir / file_name))
        elif node.IsA("vtkMRMLMarkupsNode"):
            file_name = f"{base}.fcsv"
            slicer.util.saveNode(node, str(self._work_dir / file_name))
        elif node.IsA("vtkMRMLSegmentationNode"):
            labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", f"{base}_LabelMap")
            slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(node, labelmap)
            file_name = f"{base}.mha"
            self._write_volume(labelmap, self._work_dir / file_name)
            slicer.mrmlScene.RemoveNode(labelmap)
        else:
            file_name = f"{base}.mha"
            self._write_volume(node, self._work_dir / file_name)
        return file_name

    def evaluation(self, remote_server: RemoteServer | None, devices: list[str]) -> None:
        """Evaluate the registration on the modality of the active evaluation tab."""
        if remote_server is not None:
            # The metrics read the transform this machine holds: they run here, on its CPU.
            self._update_logs("[ImpactReg] Evaluation runs on this machine, not on the server.")
            devices = []

        self.evaluation_panel.clear_metrics()

        args = ["eval", "-o", "Evaluation"]

        # The registration's transform stands in for a tab's own (identity when neither is selected).
        transform_node = self.ui.inputTransformSelector.currentNode()
        evaluation_key = self._current_evaluation_key()
        if evaluation_key is None:
            # Static fallback: reference = fixed, input = moving (image modality).
            nodes = {
                "Transform": transform_node,
                "FixedImage": self.ui.referenceVolumeSelector.currentNode(),
                "MovingImage": self.ui.inputVolumeEvaluationSelector.currentNode(),
                "Mask": self.ui.referenceMaskSelector.currentNode(),
            }
        else:
            nodes = {
                key: selector.currentNode()
                for key, (_, selector) in self._evaluation_input_selectors[evaluation_key].items()
            }
            nodes["Transform"] = nodes.get("Transform") or transform_node
        for key, node in nodes.items():
            flag = "--mask" if key.startswith("Mask") else EVALUATION_FLAGS.get(key)
            if flag is None:
                self._update_logs(
                    f"[ImpactReg] impact-reg-konfai eval takes no '{key}': run the "
                    f"'{evaluation_key.display_name}' evaluation with konfai-apps eval.",
                    False,
                )
                self.template.set_running(False)
                return
            file_name = self._export_node(node, key)
            if file_name is not None:
                args += [flag, file_name]

        if devices:
            args += ["--gpu"] + devices
        else:
            args += ["--cpu", "1"]

        def on_end_function() -> None:
            evaluation_dir = self._work_dir / "Evaluation"
            json_file = next(evaluation_dir.rglob("*.json"), None)
            if json_file is None:
                self._update_logs(
                    "[ImpactReg] Evaluation finished but produced no metrics file "
                    f"(no .json under {evaluation_dir}). "
                    "The process probably failed: check the log above for errors.",
                    False,
                )
                return

            from konfai.evaluator import Statistics

            self.evaluation_panel.set_metrics(Statistics(json_file).read())
            mha_file = next(evaluation_dir.rglob("*.mha"), None)
            if mha_file is not None:
                self.evaluation_panel.refresh_images_list(mha_file.parent)

        self.process.run("impact-reg-konfai", self._work_dir, args, on_end_function)

    def uncertainty(self, remote_server: RemoteServer | None, devices: list[str]) -> None:
        """Estimate registration uncertainty from the ensemble's stack the registration kept."""
        if remote_server is not None:
            # The spread is measured on the stack this machine holds: it runs here, on its CPU.
            self._update_logs("[ImpactReg] Uncertainty runs on this machine, not on the server.")
            devices = []

        self.uncertainty_panel.clear_metrics()

        stack = self.template.ensemble_stack
        if stack is None or not stack.exists():
            self._update_logs(
                "[ImpactReg] Uncertainty needs the ensemble's stack: register with Uncertainty checked and two or "
                "more presets, or TTA on a preset that allows it.",
                False,
            )
            self.template.set_running(False)
            return

        args = ["uncertainty", "-i", str(stack), "-o", "Uncertainty"]
        if devices:
            args += ["--gpu"] + devices
        else:
            args += ["--cpu", "1"]

        def on_end_function() -> None:
            uncertainty_dir = self._work_dir / "Uncertainty"
            map_file = next(uncertainty_dir.rglob("*.mha"), None)
            if map_file is None:
                self._update_logs(
                    "[ImpactReg] Uncertainty finished but produced no map "
                    f"(no .mha under {uncertainty_dir}). "
                    "The process probably failed: check the log above for errors.",
                    False,
                )
                return

            # The map is the spread of the displacement vectors, in mm; its mean and 95th percentile are its metrics.
            from konfai.evaluator import Statistics

            metrics = next(uncertainty_dir.rglob("Metric_*.json"), None)
            if metrics is not None:
                self.uncertainty_panel.set_metrics(Statistics(metrics).read())
            self.uncertainty_panel.refresh_images_list(map_file.parent)

        self.process.run("impact-reg-konfai", self._work_dir, args, on_end_function)


class ImpactRegAppTemplateWidget(KonfAIAppTemplateWidget):
    """Generic KonfAI app widget with the registration-specific inference and QA panels."""

    INFERENCE_PANEL_CLASS = RegistrationInferencePanel
    QA_PANEL_CLASS = RegistrationQAPanel
    #: The stack of the last registration kept under Uncertainty, which the QA panel's uncertainty measures.
    ensemble_stack: Path | None = None

    def keep_stack(self, stack: Path | None) -> None:
        """Keep the last registration's ``stack`` apart from the work dir, which the next run replaces."""
        if self.ensemble_stack is not None:
            shutil.rmtree(self.ensemble_stack.parent, ignore_errors=True)
        self.ensemble_stack = None
        if stack is not None:
            self.ensemble_stack = Path(slicer.util.tempDirectory()) / stack.name
            shutil.move(str(stack), str(self.ensemble_stack))

    def get_preset_names(self) -> list[str]:
        """The presets of the apps listed in the selection combo, as ``APP#PRESET`` (an app with none by its name)."""
        combo = self.ui.appComboBox
        names = []
        for app in (combo.itemData(i) for i in range(combo.count)):
            if app is not None:
                engine = app.get_name().split(":")[-1]
                names += [f"{engine}#{preset}" for preset in app.get_presets()] or [engine]
        return names


class ImpactRegWidget(ScriptedLoadableModuleWidget):
    """
    Top-level scripted loadable module widget for ImpactReg.

    This class ties together the Slicer module system with the KonfAICoreWidget,
    which handles actual application logic and GUI.
    """

    # Major version of the KonfAI extension API this module is written against.
    # KonfAI only bumps it on breaking changes, after a deprecation cycle.
    REQUIRED_KONFAI_API_MAJOR = 2

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        super().__init__(parent)
        self.konfai_core = None

    def setup(self) -> None:
        """
        Construct and initialize the module GUI.

        This method is called once when the user first opens the module.
        """
        super().setup()

        import KonfAI as konfai_module  # noqa: N813

        api_version = getattr(konfai_module, "KONFAI_SLICER_API_VERSION", (1, 0))
        if api_version[0] != self.REQUIRED_KONFAI_API_MAJOR:
            slicer.util.errorDisplay(
                f"ImpactReg requires the KonfAI extension API version {self.REQUIRED_KONFAI_API_MAJOR}.x, "
                f"but the installed KonfAI extension provides {api_version[0]}.{api_version[1]}.\n\n"
                "Please update the KonfAI and ImpactReg extensions together."
            )
            return

        self.konfai_core = KonfAICoreWidget("Impact Reg")
        self.konfai_core.register_apps([ImpactRegAppTemplateWidget("Registration", [IMPACT_REG_REPO])])
        self.layout.addWidget(self.konfai_core)

        if _is_reload_setup("SlicerImpactReg"):
            self.konfai_core.enter()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        if self.konfai_core is not None:
            self.konfai_core.cleanup()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        if self.konfai_core is not None:
            # The registration panels shell out to the ``impact-reg-konfai`` CLI; ensure that package is
            # installed (it pins and pulls the matching konfai / konfai-apps). Shared installer from the
            # KonfAI extension: it checks PyPI and offers the upgrade when a newer release exists.
            install_package("impact-reg-konfai", "IMPACT-Reg")
            self.konfai_core.enter()

    def exit(self) -> None:  # noqa: A003
        """
        Called each time the user navigates away from this module.
        """
        if self.konfai_core is not None:
            self.konfai_core.exit()
