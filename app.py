from __future__ import annotations

import random
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from double_peak_generator import (
    GeneratorConfig, RangeSpec,
    generate_dataset_chunk, concatenate_datasets, make_dataset_zip,
    WAVELET_TYPES, wavelet_transform_1d,
)

st.set_page_config(page_title="Double Peak Generator", layout="wide")

st.markdown("""
<style>
/* Main application width */
.block-container {
    max-width: none !important;
    width: 100% !important;
    padding-top: 1.0rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
}

/* Larger controls in the left pane only */
.dpg-left-marker ~ * {
    font-size: 1.18rem;
}

/* General inputs */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    min-height: 2.85rem;
    font-size: 1.08rem;
}

/* The two real Streamlit columns are manipulated by the splitter JS. */
div[data-testid="stHorizontalBlock"]:has(#dpg-left-marker):has(#dpg-right-marker) {
    gap: 0 !important;
    align-items: stretch !important;
}

div[data-testid="stHorizontalBlock"]:has(#dpg-left-marker):has(#dpg-right-marker)
> div[data-testid="stColumn"] {
    min-width: 0 !important;
}

/* Left panel surface */
#dpg-left-marker {
    display: none;
}
#dpg-right-marker {
    display: none;
}

/* A subtle surface is applied by JS to the left Streamlit column. */
.dpg-split-left {
    background: rgba(128,128,128,0.08);
    border-radius: 10px;
    padding: 1.2rem 1.35rem 1.5rem 1.35rem !important;
    overflow-x: hidden;
    min-width: 0;
}

/* Right pane */
.dpg-split-right {
    padding: 0.25rem 0.4rem 1rem 1.35rem !important;
    min-width: 0 !important;
}

/* Hide the old sidebar completely; it is no longer used. */
section[data-testid="stSidebar"] {
    display: none !important;
}

@media (max-width: 850px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Install a robust draggable divider inside the two-column container.
# The cleanup hook prevents stale observers/listeners from surviving Streamlit reruns.
components.html(
    r"""
<script>
(() => {
  const P = window.parent;
  const D = P.document;

  // Clean up the previous splitter instance before installing a new one.
  if (typeof P.__DPG_SPLIT_CLEANUP === "function") {
    try { P.__DPG_SPLIT_CLEANUP(); } catch (e) {}
  }

  let cleanupFns = [];
  let retryTimer = null;

  function install() {
    const lm = D.getElementById("dpg-left-marker");
    const rm = D.getElementById("dpg-right-marker");
    if (!lm || !rm) return false;

    const left = lm.closest('[data-testid="stColumn"]');
    const right = rm.closest('[data-testid="stColumn"]');
    if (!left || !right) return false;

    const row = left.parentElement;
    if (!row || right.parentElement !== row) return false;

    left.classList.add("dpg-split-left");
    right.classList.add("dpg-split-right");

    row.style.position = "relative";
    row.style.gap = "0";

    let pct = Number(P.__DPG_SPLIT_PCT || 62);
    pct = Math.min(78, Math.max(28, pct));

    function apply(p) {
      pct = Math.min(78, Math.max(28, p));
      P.__DPG_SPLIT_PCT = pct;

      // Both columns remain inside the same flex row.
      left.style.flex = `0 0 ${pct}%`;
      left.style.width = `${pct}%`;
      left.style.maxWidth = `${pct}%`;
      left.style.minWidth = "0";

      right.style.flex = `0 0 ${100 - pct}%`;
      right.style.width = `${100 - pct}%`;
      right.style.maxWidth = `${100 - pct}%`;
      right.style.minWidth = "0";
    }

    apply(pct);

    // Remove any orphaned divider from a previous render.
    D.querySelectorAll("#dpg-split-divider").forEach(el => el.remove());

    const divider = D.createElement("div");
    divider.id = "dpg-split-divider";
    divider.setAttribute("role", "separator");
    divider.setAttribute("aria-orientation", "vertical");
    divider.setAttribute("aria-label", "Resize generator and results panels");
    divider.title = "Drag to resize panels";

    Object.assign(divider.style, {
      position: "absolute",
      top: "0",
      bottom: "0",
      width: "14px",
      marginLeft: "-7px",
      cursor: "col-resize",
      zIndex: "30",
      background: "transparent",
      touchAction: "none",
      userSelect: "none"
    });

    const line = D.createElement("div");
    Object.assign(line.style, {
      position: "absolute",
      left: "6px",
      top: "0",
      bottom: "0",
      width: "2px",
      background: "rgba(120,120,120,0.35)",
      borderRadius: "2px",
      transition: "background 80ms ease"
    });
    divider.appendChild(line);
    row.appendChild(divider);

    function placeDivider() {
      if (!row.isConnected || !left.isConnected || !right.isConnected) return;
      divider.style.left = `${pct}%`;
    }
    placeDivider();

    let dragging = false;

    function onPointerDown(ev) {
      dragging = true;
      line.style.background = "rgba(90,90,90,0.8)";
      try { divider.setPointerCapture(ev.pointerId); } catch (e) {}
      ev.preventDefault();
    }

    function onPointerMove(ev) {
      if (!dragging) return;
      const rect = row.getBoundingClientRect();
      if (rect.width <= 0) return;

      const raw = ((ev.clientX - rect.left) / rect.width) * 100;
      apply(raw);
      placeDivider();
      ev.preventDefault();
    }

    function onPointerUp(ev) {
      if (!dragging) return;
      dragging = false;
      line.style.background = "rgba(120,120,120,0.35)";
      try { divider.releasePointerCapture(ev.pointerId); } catch (e) {}
    }

    divider.addEventListener("pointerdown", onPointerDown);
    divider.addEventListener("pointermove", onPointerMove);
    divider.addEventListener("pointerup", onPointerUp);
    divider.addEventListener("pointercancel", onPointerUp);

    cleanupFns.push(() => divider.removeEventListener("pointerdown", onPointerDown));
    cleanupFns.push(() => divider.removeEventListener("pointermove", onPointerMove));
    cleanupFns.push(() => divider.removeEventListener("pointerup", onPointerUp));
    cleanupFns.push(() => divider.removeEventListener("pointercancel", onPointerUp));

    const resizeObserver = new ResizeObserver(() => placeDivider());
    resizeObserver.observe(row);
    cleanupFns.push(() => resizeObserver.disconnect());

    // If Streamlit replaces the row during a rerun, remove this instance cleanly.
    const bodyObserver = new MutationObserver(() => {
      if (!row.isConnected) {
        if (typeof P.__DPG_SPLIT_CLEANUP === "function") {
          P.__DPG_SPLIT_CLEANUP();
        }
      }
    });
    bodyObserver.observe(D.body, {childList: true, subtree: true});
    cleanupFns.push(() => bodyObserver.disconnect());

    P.__DPG_SPLIT_CLEANUP = () => {
      cleanupFns.forEach(fn => {
        try { fn(); } catch (e) {}
      });
      cleanupFns = [];
      try { divider.remove(); } catch (e) {}

      // Clear only styles/classes we own on this exact render.
      try {
        left.classList.remove("dpg-split-left");
        right.classList.remove("dpg-split-right");
        left.style.flex = "";
        left.style.width = "";
        left.style.maxWidth = "";
        left.style.minWidth = "";
        right.style.flex = "";
        right.style.width = "";
        right.style.maxWidth = "";
        right.style.minWidth = "";
      } catch (e) {}
    };

    return true;
  }

  let tries = 0;
  retryTimer = setInterval(() => {
    tries += 1;
    if (install() || tries > 80) {
      clearInterval(retryTimer);
      retryTimer = null;
    }
  }, 100);

  // If the iframe itself is destroyed, remove timers and splitter listeners.
  window.addEventListener("beforeunload", () => {
    if (retryTimer) clearInterval(retryTimer);
  });
})();
</script>
""",
    height=0,
    scrolling=False,
)

st.title("Double Peak Generator")
st.caption("Generate two-component mathematical peak datasets or the locked CA-FA legacy preset.")

defaults = {
    "dataset": None,
    "preview_indices": [],
    "generation_parts": [],
    "generation_config": None,
    "generation_target": 0,
    "generation_done": 0,
    "generation_running": False,
    "generation_paused": False,
    "generation_finished": False,
    "generation_error": None,
    "file_name_state": "double_peak_dataset",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def rr(label, default_min, default_max, key_prefix, disabled=False):
    c1, c2 = st.columns(2)
    with c1:
        lo = st.number_input(
            f"{label} min", value=float(default_min),
            key=f"{key_prefix}_min", disabled=disabled
        )
    with c2:
        hi = st.number_input(
            f"{label} max", value=float(default_max),
            key=f"{key_prefix}_max", disabled=disabled
        )
    return RangeSpec(float(lo), float(hi))

left, right = st.columns([62, 38], gap=None)

with left:
    st.markdown('<div id="dpg-left-marker"></div>', unsafe_allow_html=True)

    st.header("Generator settings")

    st.subheader("Generator mode")
    cafa_mode = st.checkbox(
        "CA-FA",
        value=False,
        help="Locked legacy preset. Disables all custom mathematical controls."
    )
    mode = "cafa" if cafa_mode else "custom"
    locked = cafa_mode

    if cafa_mode:
        st.info("CA-FA mode uses the locked legacy generator settings. Custom controls are disabled.")

    st.subheader("Mathematical generation mode")
    strategy_label = st.radio(
        "Generation strategy",
        [
            "1. Independent parameter ranges",
            "2. Width-constrained geometry",
            "3. Width-constrained mixed G/L/PV (1/3 each)",
        ],
        index=0,
        disabled=locked,
    )
    if strategy_label.startswith("1."):
        generation_strategy = "independent"
    elif strategy_label.startswith("2."):
        generation_strategy = "constrained"
    else:
        generation_strategy = "constrained_mix"

    st.subheader("Signal axis")
    # CA-FA keeps the exact legacy axis visible but locked. Custom modes start
    # with a compact, easy-to-inspect 352-point example range.
    default_x_min = 400.0 if cafa_mode else 55.0
    default_x_max = 1102.0 if cafa_mode else 200.0
    default_n_points = 352
    x1, x2 = st.columns(2)
    with x1:
        x_min = st.number_input("X min", value=default_x_min, disabled=locked, key="cafa_x_min" if cafa_mode else "custom_x_min")
    with x2:
        x_max = st.number_input("X max", value=default_x_max, disabled=locked, key="cafa_x_max" if cafa_mode else "custom_x_max")

    min_points_ui = 201 if generation_strategy in {"constrained", "constrained_mix"} else 2
    n_points = st.number_input(
        "Number of points",
        min_value=min_points_ui,
        value=max(default_n_points, min_points_ui),
        step=1,
        disabled=locked,
        key="cafa_n_points" if cafa_mode else "custom_n_points"
    )
    if not locked and n_points >= 2 and x_max > x_min:
        st.caption(f"Δx = {(x_max-x_min)/(n_points-1):.6g}")

    if generation_strategy == "independent":
        st.subheader("Peak parameters")
        if cafa_mode:
            # Parameter ranges implied by c in [1, 10]; shown for transparency only.
            mu1 = rr("μ1 (CA)", 755.866667, 758.641664, "cafa_mu1", True)
            mu2 = rr("μ2 (FA)", 769.139286, 785.5, "cafa_mu2", True)
            sigma1 = rr("σ1 (CA)", 96.037868, 119.412398, "cafa_sigma1", True)
            sigma2 = rr("σ2 (FA)", 84.343311, 122.187848, "cafa_sigma2", True)
            A1 = rr("A1 (CA)", 0.05048413, 0.22744591, "cafa_A1", True)
            A2 = rr("A2 (FA)", 0.07968741, 0.30205464, "cafa_A2", True)
            st.caption("CA and FA concentrations are sampled independently from 1 to 10; these locked ranges are the resulting parameter spans.")
        else:
            mu1 = rr("μ1", 100, 110, "mu1", False)
            mu2 = rr("μ2", 130, 140, "mu2", False)
            sigma1 = rr("σ1", 12, 18, "sigma1", False)
            sigma2 = rr("σ2", 12, 18, "sigma2", False)
            A1 = rr("A1", 0.1, 1.0, "A1", False)
            A2 = rr("A2", 0.1, 1.0, "A2", False)

        st.subheader("Peak type")
        pc1, pc2 = st.columns(2)
        with pc1:
            gaussian = st.checkbox("Gaussian", value=True, disabled=locked)
            skew = st.checkbox("Skew Normal", value=False, disabled=locked)
        with pc2:
            lorentz = st.checkbox("Lorentzian", value=False, disabled=locked)
            pv = st.checkbox("Pseudo-Voigt", value=False, disabled=locked)

        selected = []
        if gaussian: selected.append("Gaussian")
        if skew: selected.append("Skew Normal")
        if lorentz: selected.append("Lorentzian")
        if pv: selected.append("Pseudo-Voigt")

        paired_peak_types = st.checkbox(
            "Sample peak types in pairs",
            value=True,
            disabled=locked,
            help=(
                "If enabled, one shape family is sampled for each generated pair, "
                "so both peaks share it (e.g. Gaussian+Gaussian or Pseudo-Voigt+Pseudo-Voigt). "
                "If disabled, Peak 1 and Peak 2 sample their types independently."
            ),
        )

        with st.expander("Additional shape parameters", expanded=skew or pv):
            alpha1 = rr("α1", -5, 5, "alpha1", locked or not skew)
            alpha2 = rr("α2", -5, 5, "alpha2", locked or not skew)
            eta1 = rr("η1", 0, 1, "eta1", locked or not pv)
            eta2 = rr("η2", 0, 1, "eta2", locked or not pv)

        constrained_shape = "Gaussian"
        min_peak_distance_points = 0.0
        ampmax = RangeSpec(0.5, 1.0)
        ampratio = RangeSpec(0.1, 1.0)

    else:
        mu1 = RangeSpec(0.0, 0.0)
        mu2 = RangeSpec(0.0, 0.0)
        sigma1 = RangeSpec(1.0, 1.0)
        sigma2 = RangeSpec(1.0, 1.0)
        # In constrained modes amplitudes are sampled independently.
        A1 = RangeSpec(0.1, 1.0)
        A2 = RangeSpec(0.1, 1.0)
        alpha1 = RangeSpec(0.0, 0.0)
        alpha2 = RangeSpec(0.0, 0.0)
        selected = ("Gaussian",)

        st.subheader("Width-constrained geometry")
        st.caption("σ is sampled in point units: 0.02n–0.06n; 0.5 ≤ σ1/σ2 ≤ 2.")

        if generation_strategy == "constrained":
            st.subheader("Peak type")
            pc1, pc2 = st.columns(2)
            with pc1:
                gaussian_c = st.checkbox("Gaussian", value=True, disabled=locked, key="c_gaussian")
                lorentz_c = st.checkbox("Lorentzian", value=False, disabled=locked, key="c_lorentz")
            with pc2:
                pv_c = st.checkbox("Pseudo-Voigt", value=False, disabled=locked, key="c_pv")
            selected = []
            if gaussian_c: selected.append("Gaussian")
            if lorentz_c: selected.append("Lorentzian")
            if pv_c: selected.append("Pseudo-Voigt")
            paired_peak_types = st.checkbox(
                "Sample peak types in pairs",
                value=True,
                disabled=locked,
                key="paired_constrained",
                help=(
                    "Enabled: both peaks in a sample share one randomly selected active shape family. "
                    "Disabled: the two peak types are sampled independently from the active families."
                ),
            )
            constrained_shape = selected[0] if selected else "Gaussian"
        else:
            constrained_shape = "Gaussian"
            selected = ["Gaussian", "Lorentzian", "Pseudo-Voigt"]
            paired_peak_types = st.checkbox(
                "Sample peak types in pairs",
                value=False,
                disabled=locked,
                key="paired_constrained_mix",
                help=(
                    "Enabled: one family is sampled with probability 1/3 from Gaussian, Lorentzian, "
                    "or Pseudo-Voigt and both peaks use it. Disabled: Peak 1 and Peak 2 each choose "
                    "their family independently with probability 1/3."
                ),
            )
            if paired_peak_types:
                st.info(
                    "One of Gaussian, Lorentzian or Pseudo-Voigt is sampled with probability 1/3, "
                    "and both peaks in the pair use that same family."
                )
            else:
                st.info(
                    "Peak 1 and Peak 2 independently choose Gaussian, Lorentzian or Pseudo-Voigt "
                    "with probability 1/3 each."
                )

        min_peak_distance_points = st.number_input(
            "Minimum peak distance [points]",
            min_value=0.0,
            value=10.0,
            step=1.0,
            disabled=locked,
        )
        st.caption("Δμ is sampled from max(0.25·w, minimum distance) to 3·w.")

        st.subheader("Amplitude ranges")
        A1 = rr("A1", 0.1, 1.0, "A1_constrained", locked)
        A2 = rr("A2", 0.1, 1.0, "A2_constrained", locked)
        st.caption("A1 and A2 are sampled independently. Both may be small, medium, large or equal.")

        # Kept only for backward-compatible configuration fields; ignored by constrained generation.
        ampmax = RangeSpec(0.5, 1.0)
        ampratio = RangeSpec(0.1, 1.0)

        st.subheader("Pseudo-Voigt parameter")
        eta1 = rr("η1", 0.0, 1.0, "eta1_constrained", locked)
        eta2 = rr("η2", 0.0, 1.0, "eta2_constrained", locked)

    st.subheader("Noise")
    noise_enabled = st.checkbox("Enable noise", value=True, disabled=locked)
    if cafa_mode:
        noise_pct = rr("Noise [%]", 1, 1, "cafa_noise", True)
        st.caption("Legacy CA-FA noise is fixed at exactly 1% peak-to-peak of each individual peak height.")
    else:
        noise_pct = rr("Noise [%]", 1, 1, "noise", not noise_enabled)

    st.subheader("Dataset")
    d1, d2 = st.columns(2)
    with d1:
        n_samples = st.number_input(
            "Number of samples",
            min_value=1,
            max_value=100000,
            value=1000,
            step=1
        )
    with d2:
        seed = st.number_input("Seed", min_value=0, value=20260925, step=1)

    file_name = st.text_input("File name", value=st.session_state.file_name_state)
    st.session_state.file_name_state = file_name

    start_clicked = st.button("Generate", type="primary", use_container_width=True)

with right:
    st.markdown('<div id="dpg-right-marker"></div>', unsafe_allow_html=True)

    st.header("Generation control")

    progress = 0.0
    if st.session_state.generation_target:
        progress = min(
            1.0,
            st.session_state.generation_done / st.session_state.generation_target
        )
    progress_bar = st.progress(progress)
    progress_text = st.empty()
    progress_text.write(
        f"{st.session_state.generation_done:,} / "
        f"{st.session_state.generation_target:,} samples"
        if st.session_state.generation_target
        else "Ready"
    )

    c1, c2 = st.columns(2)
    with c1:
        pause_clicked = st.button(
            "Pause",
            disabled=not st.session_state.generation_running,
            use_container_width=True,
        )
    with c2:
        continue_clicked = st.button(
            "Continue",
            disabled=not st.session_state.generation_paused,
            use_container_width=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        reset_clicked = st.button(
            "Reset",
            disabled=not (
                st.session_state.generation_running
                or st.session_state.generation_paused
                or st.session_state.generation_done > 0
            ),
            use_container_width=True,
        )
    with c4:
        partial_available = bool(st.session_state.generation_parts)
        save_partial_clicked = st.button(
            "Prepare partial ZIP",
            disabled=not partial_available,
            use_container_width=True,
        )

    if reset_clicked:
        st.session_state.dataset = None
        st.session_state.preview_indices = []
        st.session_state.generation_parts = []
        st.session_state.generation_config = None
        st.session_state.generation_target = 0
        st.session_state.generation_done = 0
        st.session_state.generation_running = False
        st.session_state.generation_paused = False
        st.session_state.generation_finished = False
        st.session_state.generation_error = None
        st.rerun()

    if pause_clicked:
        st.session_state.generation_running = False
        st.session_state.generation_paused = True

    if continue_clicked:
        st.session_state.generation_running = True
        st.session_state.generation_paused = False

    if start_clicked:
        try:
            cfg = GeneratorConfig(
                mode=mode,
                x_min=float(x_min),
                x_max=float(x_max),
                n_points=int(n_points),
                mu1=mu1, mu2=mu2,
                sigma1=sigma1, sigma2=sigma2,
                A1=A1, A2=A2,
                alpha1=alpha1, alpha2=alpha2,
                eta1=eta1, eta2=eta2,
                selected_peak_types=tuple(selected),
                paired_peak_types=bool(paired_peak_types),
                noise_enabled=bool(noise_enabled),
                noise_pct=noise_pct,
                n_samples=int(n_samples),
                seed=int(seed),
                generation_strategy=generation_strategy,
                constrained_shape=constrained_shape,
                min_peak_distance_points=float(min_peak_distance_points),
                amplitude_max_min=float(ampmax.min),
                amplitude_max_max=float(ampmax.max),
                amplitude_ratio_min=float(ampratio.min),
                amplitude_ratio_max=float(ampratio.max),
            )
            cfg.validate()
            st.session_state.dataset = None
            st.session_state.preview_indices = []
            st.session_state.generation_parts = []
            st.session_state.generation_config = cfg
            st.session_state.generation_target = int(n_samples)
            st.session_state.generation_done = 0
            st.session_state.generation_running = True
            st.session_state.generation_paused = False
            st.session_state.generation_finished = False
            st.session_state.generation_error = None
        except Exception as exc:
            st.session_state.generation_error = str(exc)

    if st.session_state.generation_error:
        st.error(st.session_state.generation_error)

    if st.session_state.generation_running and st.session_state.generation_config is not None:
        cfg = st.session_state.generation_config
        remaining = st.session_state.generation_target - st.session_state.generation_done

        if remaining > 0:
            chunk_size = min(500, remaining)
            start = st.session_state.generation_done

            try:
                part = generate_dataset_chunk(cfg, start=start, count=chunk_size)
                st.session_state.generation_parts.append(part)
                st.session_state.generation_done += chunk_size

                frac = (
                    st.session_state.generation_done
                    / st.session_state.generation_target
                )
                progress_bar.progress(min(1.0, frac))
                progress_text.write(
                    f"{st.session_state.generation_done:,} / "
                    f"{st.session_state.generation_target:,} samples"
                )
            except Exception as exc:
                st.session_state.generation_running = False
                st.session_state.generation_error = str(exc)
                st.error(str(exc))
            else:
                if (
                    st.session_state.generation_done
                    >= st.session_state.generation_target
                ):
                    final_ds = concatenate_datasets(
                        st.session_state.generation_parts,
                        full_config=st.session_state.generation_parts[0].config,
                    )
                    st.session_state.dataset = final_ds
                    st.session_state.generation_running = False
                    st.session_state.generation_paused = False
                    st.session_state.generation_finished = True

                    k = min(6, len(final_ds.summed))
                    rng = np.random.default_rng(int(cfg.seed) + 999)
                    st.session_state.preview_indices = list(
                        map(
                            int,
                            rng.choice(
                                len(final_ds.summed),
                                size=k,
                                replace=False
                            )
                        )
                    )
                    st.success(
                        f"Generated {len(final_ds.summed):,} samples × "
                        f"{len(final_ds.x):,} points."
                    )
                else:
                    time.sleep(0.03)
                    st.rerun()

    current_partial = None
    if st.session_state.generation_parts:
        try:
            current_partial = concatenate_datasets(
                st.session_state.generation_parts,
                full_config=st.session_state.generation_parts[0].config,
            )
        except Exception:
            current_partial = None

    if st.session_state.generation_paused:
        st.warning(
            f"Generation paused at {st.session_state.generation_done:,} / "
            f"{st.session_state.generation_target:,} samples."
        )

    if save_partial_clicked and current_partial is not None:
        st.session_state.dataset = current_partial
        k = min(6, len(current_partial.summed))
        rng = np.random.default_rng(12345 + len(current_partial.summed))
        st.session_state.preview_indices = list(
            map(
                int,
                rng.choice(
                    len(current_partial.summed),
                    size=k,
                    replace=False
                )
            )
        )
        st.info(
            "Partial dataset prepared below. "
            "You can download it without losing the paused state."
        )

    ds = st.session_state.dataset

    if ds is not None:
        suffix = "_partial" if (
            st.session_state.generation_done
            < st.session_state.generation_target
            and st.session_state.generation_target > 0
        ) else ""

        zip_bytes = make_dataset_zip(ds, file_name + suffix)

        st.download_button(
            "Download ZIP",
            data=zip_bytes,
            file_name=f"{file_name}{suffix}.zip",
            mime="application/zip",
            use_container_width=True,
            help="Your browser controls the final local save location.",
        )

        st.subheader("Random preview")

        # Interactive wavelet overlay for the generated examples.  This is
        # intentionally a preview-only control: changing it never regenerates
        # or modifies the dataset.
        st.markdown("**Wavelet preview**")
        wc1, wc2 = st.columns(2)
        with wc1:
            preview_wavelet = st.selectbox(
                "Wavelet",
                options=("None",) + tuple(WAVELET_TYPES),
                index=0,
                help=(
                    "GD1 = first Gaussian derivative; MH / GD2 = Mexican Hat "
                    "(second Gaussian derivative); GD3 = third Gaussian derivative."
                ),
            )
        with wc2:
            preview_scale_text = st.text_input(
                "Scale",
                value="",
                placeholder="1",
                help="Positive scale in sample points. Blank means scale = 1. Fractional values are allowed; both 0.5 and 0,5 are accepted.",
            )

        wc3, wc4 = st.columns(2)
        with wc3:
            preview_source = st.selectbox(
                "Transform",
                options=("Sum", "Peak 1", "Peak 2"),
                index=0,
                help="Choose which generated curve is transformed for the overlay.",
            )
        with wc4:
            normalize_overlay = st.checkbox(
                "Normalize overlay",
                value=True,
                help=(
                    "Rescale the wavelet curve to the displayed source amplitude. "
                    "Turn this off to view the true wavelet coefficient scale on a second y-axis."
                ),
            )

        preview_scale = 1.0
        preview_scale_error = None
        if preview_scale_text.strip():
            try:
                preview_scale_normalized = preview_scale_text.strip().replace(",", ".")
                preview_scale = float(preview_scale_normalized)
                if not np.isfinite(preview_scale) or preview_scale <= 0:
                    raise ValueError
            except ValueError:
                preview_scale_error = "Scale must be a finite positive number. Blank means scale = 1. Fractional values may use either a dot or a comma."
                st.warning(preview_scale_error)

        if preview_wavelet != "None" and preview_scale_error is None:
            st.caption(
                f"Overlay: {preview_wavelet}, scale = {preview_scale:g}, source = {preview_source}. "
                "Changing these controls updates the preview only."
            )

        if st.button("Draw another 6", use_container_width=True):
            k = min(6, len(ds.summed))
            st.session_state.preview_indices = random.sample(
                range(len(ds.summed)),
                k
            )

        st.caption(
            "These are actual samples selected from the generated dataset."
        )

        idxs = st.session_state.preview_indices

        # In a narrower right pane, use one preview per row.
        # This naturally expands as the user drags the separator left.
        for idx in idxs:
            r = ds.labels.iloc[idx]

            fig, ax = plt.subplots(figsize=(6.8, 3.5))
            ax.plot(
                ds.x, ds.peak1[idx],
                label="Peak 1",
                linewidth=1.3
            )
            ax.plot(
                ds.x, ds.peak2[idx],
                label="Peak 2",
                linewidth=1.3
            )
            ax.plot(
                ds.x, ds.summed[idx],
                label="Peak 1 + Peak 2",
                linewidth=1.6
            )

            wavelet_ax = None
            if preview_wavelet != "None" and preview_scale_error is None:
                source_map = {
                    "Sum": ds.summed[idx],
                    "Peak 1": ds.peak1[idx],
                    "Peak 2": ds.peak2[idx],
                }
                source_y = np.asarray(source_map[preview_source], dtype=float)
                coeff = wavelet_transform_1d(
                    source_y, preview_wavelet, preview_scale
                )

                if normalize_overlay:
                    # Preserve coefficient sign while matching the displayed
                    # source amplitude, so shape/localization can be compared
                    # directly on the same axes.
                    cmax = float(np.max(np.abs(coeff)))
                    smax = float(np.max(np.abs(source_y)))
                    overlay = coeff if cmax <= 0 else coeff * (smax / cmax)
                    ax.plot(
                        ds.x, overlay,
                        linestyle="--", linewidth=1.4,
                        label=f"{preview_wavelet}, s={preview_scale:g} ({preview_source})"
                    )
                else:
                    wavelet_ax = ax.twinx()
                    wavelet_ax.plot(
                        ds.x, coeff,
                        linestyle="--", linewidth=1.25,
                        label=f"{preview_wavelet}, s={preview_scale:g} ({preview_source})"
                    )
                    wavelet_ax.set_ylabel("wavelet coefficient")

            ax.set_title(f"Sample {idx}")
            ax.set_xlabel("x")
            ax.set_ylabel("signal")

            handles, labels = ax.get_legend_handles_labels()
            if wavelet_ax is not None:
                h2, l2 = wavelet_ax.get_legend_handles_labels()
                handles += h2
                labels += l2
            ax.legend(handles, labels, fontsize=8)
            ax.grid(alpha=0.2)
            st.pyplot(fig, clear_figure=True, use_container_width=True)

            p1_extra = ""
            if r["peak1_type"] == "Skew Normal":
                p1_extra = f", α={r['alpha1']:.4g}"
            elif r["peak1_type"] == "Pseudo-Voigt":
                p1_extra = f", η={r['eta1']:.4g}"

            p2_extra = ""
            if r["peak2_type"] == "Skew Normal":
                p2_extra = f", α={r['alpha2']:.4g}"
            elif r["peak2_type"] == "Pseudo-Voigt":
                p2_extra = f", η={r['eta2']:.4g}"

            st.markdown(
                f"""**Peak 1:** {r['peak1_type']}  
μ={r['mu1']:.4g}, σ={r['sigma1']:.4g}, A={r['A1']:.4g}{p1_extra}  
**Peak 2:** {r['peak2_type']}  
μ={r['mu2']:.4g}, σ={r['sigma2']:.4g}, A={r['A2']:.4g}{p2_extra}  
**Noise:** {r['noise_pct1']:.3g}% / {r['noise_pct2']:.3g}%"""
            )

        st.subheader("Pipeline hand-off")
        st.code(
            "payload = dataset.pipeline_payload()\n"
            "X = payload['spectra']\n"
            "labels = payload['labels']\n"
            "x_axis = payload['x_axis']",
            language="python",
        )

    with st.expander("Help"):
        st.markdown(r"""
### How the generator works

**Quick-start defaults**  
Custom modes start with `X min = 55`, `X max = 200` and `352` points. Increasing the number of points (for example 704 or 1408) increases sampling density while keeping the same X range; there are no hidden 2× or 4× multipliers. Gaussian is the initially selected peak family where shape selection is available. In Modes 2 and 3, the initial minimum peak distance is `10` points. Custom noise starts at a fixed `1%` (`min = max = 1`). Paired type sampling starts enabled in Modes 1 and 2; users can turn it off whenever independently mixed peak families are desired. Mode 3 also offers the same checkbox; there it starts disabled to preserve the original independent 1/3–1/3–1/3 mixed preset.

**CA-FA preset**  
CA-FA is the locked legacy preset. Its settings remain visible but inactive: X = 400–1102 mV, step = 2 mV, 352 points, Gaussian peaks, independent CA and FA concentrations from 1 to 10, and exactly 1% peak-to-peak noise added separately to each component before summation. The locked μ, σ and A boxes show the parameter ranges implied by the CA/FA concentration equations.

**Wavelet preview**  
The right-hand preview can overlay a Gaussian-derivative wavelet transform on any generated example. `GD1` emphasizes slope/edge changes; `MH` / `GD2` emphasizes curvature and local maxima; `GD3` is more sensitive to subtle shape changes and also to noise. Enter any positive scale in sample points; fractional values such as `0.5`, `0.75`, `1.5` or `3.7` are allowed, and the field accepts either a decimal dot or comma. A blank Scale field means `scale = 1`. Very small scales may contain too few discrete samples to be informative. `Transform` selects Sum, Peak 1, or Peak 2. With `Normalize overlay` enabled, coefficients are rescaled only for visual comparison; disabling it shows the true coefficient values on a second y-axis. Wavelet controls change the preview only and never alter the generated dataset.

**Axis**  
`X min` and `X max` define the complete mathematical signal range. `Number of points` creates evenly spaced points including both endpoints. For a single Gaussian, the visually useful ±3σ interval is `[μ - 3σ, μ + 3σ]`; for a pair, a convenient full view is approximately `[μ1 - 3σ1, μ2 + 3σ2]`.

**Main parameters**  
Each peak has `μ`, `σ`, and `A`. Every range is entered as `min` and `max`. If `min = max`, the value is constant. If `min < max`, it is sampled uniformly.

**Mode 1 — Independent parameter ranges**  
μ, σ and A are sampled independently from user-defined ranges. Select one or more mathematical peak families. With `Sample peak types in pairs` enabled, one active family is sampled per example and both peaks use it (for example Gaussian+Gaussian or Pseudo-Voigt+Pseudo-Voigt). With the checkbox disabled, Peak 1 and Peak 2 sample their shape families independently, so mixed pairs are allowed. If only one family is selected, the checkbox naturally makes no difference.

**Mode 2 — Width-constrained geometry**  
For n > 200: σ1 and σ2 are sampled from 0.02n–0.06n, 0.5 ≤ σ1/σ2 ≤ 2, FWHM = 2.355σ, w = (FWHM1+FWHM2)/2, and Δμ is sampled from max(0.25w, minimum distance) to 3w. The pair center is sampled so both peaks remain at least 4σ from the signal edges. Select any combination of Gaussian, Lorentzian and Pseudo-Voigt. `Sample peak types in pairs` has the same meaning as in Mode 1: enabled = same sampled family inside a pair; disabled = independent shape sampling for Peak 1 and Peak 2.

**Mode 3 — Width-constrained mixed G/L/PV**  
Uses the same geometric constraints as Mode 2, but the available families are fixed to Gaussian, Lorentzian and Pseudo-Voigt. With `Sample peak types in pairs` disabled, Peak 1 and Peak 2 independently choose one of these three families with probability 1/3 each. With the checkbox enabled, one of the three families is sampled with probability 1/3 and both peaks in the pair use that same family.

**Noise**  
Noise is generated independently for Peak 1 and Peak 2. A normal random vector is scaled so its peak-to-peak span equals the sampled percentage of that individual peak height. The custom GUI starts at a fixed 1% (`Noise min = Noise max = 1`), but the user can freely change the range. `Noise min = Noise max = 2` means exactly 2% for every peak; `0–2` means the percentage is sampled independently from that interval. CA-FA is fixed at exactly 1%.

**Pause / Continue / Reset**  
Generation is performed in chunks so the GUI remains controllable. `Pause` stops after the current chunk. `Continue` resumes from the next chunk. `Reset` discards the current run. `Prepare partial ZIP` lets you download the samples already generated.

**Resizable layout**  
Drag the vertical separator between the settings panel and the results panel. Moving it left makes the settings panel smaller and the results panel larger; moving it right does the opposite.
""")
