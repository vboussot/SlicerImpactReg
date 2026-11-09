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
``impact-reg-konfai register`` with the presets of the base panel's chips (several are
ensembled: their displacement fields averaged, their stack kept with Uncertainty), and
evaluation runs ``impact-reg-konfai eval`` (the CLI warps the moving data with the
transform). Uncertainty is the base panel's: konfai-apps, with the engine apps' tabs.
"""

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
from qt import QWidget
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget

IMPACT_REG_REPO = "VBoussot/ImpactReg"

# Map each evaluation config of the engine apps to the fixed/moving flags of `impact-reg-konfai eval`: in the
# declared order, the Transform entry is --transform, then the fixed side, the moving side, and Mask_* the mask.
# A config impact-reg-konfai does not run as a tab runs as any app's evaluation, konfai-apps eval: the inverse
# consistency (a second transform), and the measures without reference (the transform, and the fixed image a
# B-spline is sampled on).
EVALUATION_FLAGS = {
    "Evaluation_with_images.yml": ("-f", "-m"),
    "Evaluation_with_seg.yml": ("--gt-fixed-seg", "--gt-moving-seg"),
    "Evaluation_with_fid.yml": ("--gt-fixed-fid", "--gt-moving-fid"),
}


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

    The presets are the base panel's (its Presets chips, ``APP#PRESET`` of any engine): several are ensembled, their
    displacement fields averaged into one transform, and with Uncertainty their stack kept for the QA tab.
    """

    def __init__(self, template) -> None:
        super().__init__(template)
        self.ui.uncertaintyCheckBox.toolTip = _(
            "Keep the ensemble's stack, every preset's displacement field, so the QA tab can measure how they "
            "disagree."
        )

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
        presets = [reference.split(":")[-1] for reference in self.references()]  # register names them in its repo
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
        # The Uncertainty checkbox keeps the ensemble's stack (every preset's displacement field) beside the averaged
        # transform, so the QA panel can measure how the presets disagree.
        if self.ui.uncertaintyCheckBox.isChecked():
            args += ["--uncertainty"]
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

            # The ensemble's stack, with --uncertainty: one sequence item per preset, its displacement field.
            stack = next((path for path in sorted(case_dir.glob("InferenceStack.*")) if path.is_file()), None)
            if stack is not None:
                self.load_stack(stack, presets)
            else:
                self.clear_stack()  # none of the previous registration's fields left to measure

            self._update_logs("Registration results loaded into Slicer.")

        self.process.run("impact-reg-konfai", self._work_dir, args, on_end_function)


class RegistrationQAPanel(KonfAIAppQAPanel):
    """QA panel variant that drives ``impact-reg-konfai eval``; the uncertainty tabs (the engine apps' own
    Uncertainty*.yml) run as any app's, through konfai-apps.

    Unlike the generic panel, the raw (unwarped) nodes are exported: the CLI applies
    the transform to the moving data itself (images/segmentations/landmarks).
    """

    def _export_node(self, node, base: str) -> str | None:
        """Export a node for the CLI: markups as .fcsv, segmentations as labelmap .mha, volumes as .mha."""
        if node is None:
            return None
        if node.IsA("vtkMRMLMarkupsNode"):
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

        # The transform produced by the registration (identity when none is selected).
        transform_node = self.ui.inputTransformSelector.currentNode()
        if transform_node is not None:
            slicer.util.saveNode(transform_node, str(self._work_dir / "Transform.h5"))
            args += ["--transform", "Transform.h5"]

        evaluation_key = self._current_evaluation_key()
        if evaluation_key is None:
            # Static fallback: reference = fixed, input = moving (image modality).
            fixed_flag, moving_flag = EVALUATION_FLAGS["Evaluation_with_images.yml"]
            fixed_file = self._export_node(self.ui.referenceVolumeSelector.currentNode(), "FixedImage")
            moving_file = self._export_node(self.ui.inputVolumeEvaluationSelector.currentNode(), "MovingImage")
            args += [fixed_flag, fixed_file, moving_flag, moving_file]
            mask_file = self._export_node(self.ui.referenceMaskSelector.currentNode(), "Mask")
            if mask_file:
                args += ["--mask", mask_file]
        else:
            flags = EVALUATION_FLAGS.get(evaluation_key.evaluation_file)
            if flags is None:
                super().evaluation(None, devices)  # here too, as the log above says
                return
            # Manifest order: the Transform entry (it replaces the panel's transform), the fixed side, the moving
            # side, Mask_* the evaluation mask.
            sides = iter(flags)
            for key, (_required, selector) in self._evaluation_input_selectors[evaluation_key].items():
                node = selector.currentNode()
                if node is not None and node.IsA("vtkMRMLTransformNode"):
                    slicer.util.saveNode(node, str(self._work_dir / "Transform.h5"))
                    if "--transform" not in args:
                        args += ["--transform", "Transform.h5"]
                    continue
                file_name = self._export_node(node, key)
                if file_name is None:
                    continue
                if key.startswith("Mask"):
                    args += ["--mask", file_name]
                else:
                    flag = next(sides, None)
                    if flag is not None:
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


class ImpactRegAppTemplateWidget(KonfAIAppTemplateWidget):
    """Generic KonfAI app widget with the registration-specific inference and QA panels."""

    INFERENCE_PANEL_CLASS = RegistrationInferencePanel
    QA_PANEL_CLASS = RegistrationQAPanel


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
