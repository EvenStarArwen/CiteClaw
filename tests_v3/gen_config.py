"""Generate one config file per (scenario, model).

Usage:
    python tests_v3/gen_config.py <scenario_id> <model_key>
        where model_key in {gemma, grok, openai}

Writes `tests_v3/configs/<scenario_id>_<model_key>.yaml` and returns the path.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

SCENARIOS: dict[str, dict] = {
    "prime_editing": {
        "topic": textwrap.dedent("""\
            CRISPR prime editing for correcting pathogenic genetic variants —
            includes prime editor architectures (PE1/PE2/PE3/PE4/PE5/PEmax,
            twinPE, PASTE, engineered prime editing), prime editing guide RNA
            (pegRNA / epegRNA) design, mechanisms, delivery (lipid
            nanoparticle, AAV), specificity / off-targets, and therapeutic
            application to disease-causing mutations. Scope is prime editing
            specifically — NOT base editing (CBE / ABE) except where directly
            compared, NOT wild-type SpCas9 knockout / knock-in editing, NOT
            gene silencing / RNAi, NOT CRISPR screens.
        """).strip(),
    },
    "opv_flex": {
        "topic": textwrap.dedent("""\
            Organic photovoltaic (OPV) materials and devices for flexible
            solar cells — covers donor / acceptor polymers (PM6, PM7, Y6
            family, ITIC family), non-fullerene acceptors, bulk heterojunction
            morphology, flexible substrate integration, stretchability,
            stability, encapsulation, and device efficiency records specific
            to flexible cells. Scope is organic (carbon-based) PV — NOT
            perovskite solar cells, NOT silicon or thin-film inorganic PV,
            NOT dye-sensitized cells.
        """).strip(),
    },
    "spatial_txomics": {
        "topic": textwrap.dedent("""\
            Spatial transcriptomics — methods for measuring gene expression
            with preserved spatial / tissue context. Covers sequencing-based
            platforms (Visium, Slide-seq, Stereo-seq, Xenium, MERFISH,
            seqFISH, CosMx), computational methods for spatial analysis
            (cell-cell communication, spatial domains, deconvolution,
            spatially-variable gene detection), and tissue / disease atlas
            applications that develop new method. Scope is spatial
            transcriptomics methodology — NOT single-cell RNA-seq without
            spatial component, NOT bulk RNA-seq, NOT spatial proteomics.
        """).strip(),
    },
    "gw_ligo": {
        "topic": textwrap.dedent("""\
            Gravitational wave astronomy using LIGO / Virgo / KAGRA
            detectors — covers detection pipelines, compact binary coalescence
            (binary black hole, binary neutron star, neutron-star-black-hole),
            continuous and stochastic wave searches, parameter estimation
            methods, astrophysical population inference, multi-messenger
            follow-up, noise characterization, detector upgrades (A+, O4,
            Cosmic Explorer, Einstein Telescope). Scope is ground-based
            interferometric GW observation — NOT pulsar timing arrays
            (NANOGrav), NOT LISA, NOT purely theoretical GR papers without
            detector connection.
        """).strip(),
    },
    "n2_reduction": {
        "topic": textwrap.dedent("""\
            Electrocatalytic nitrogen reduction reaction (NRR) to ammonia
            under ambient conditions — catalyst design (transition metal
            single-atom, nitride, boride, alloy, 2D material), selectivity
            vs competing hydrogen evolution, faradaic efficiency,
            mechanism (associative, dissociative, MvK), in-situ /
            operando spectroscopy, electrolyte engineering, and lithium-
            mediated NRR as distinct route. Scope is electrochemical NRR
            — NOT Haber-Bosch, NOT photocatalytic N2 fixation, NOT
            biological nitrogenase computational studies.
        """).strip(),
    },
    "lnp_delivery": {
        "topic": textwrap.dedent("""\
            Lipid nanoparticle (LNP) formulations for nucleic acid delivery
            — ionizable lipid design, helper lipid / cholesterol / PEG
            composition, biodistribution, organ-targeting (SORT, endogenous
            targeting), stability, scale-up manufacturing, and applications
            to mRNA, siRNA, and CRISPR payload delivery. Scope is LNP
            chemistry / formulation / targeting — NOT viral vectors (AAV,
            lentivirus), NOT polymer / dendrimer / exosome delivery
            systems unless directly compared to LNPs.
        """).strip(),
    },
    "photocatalysis": {
        "topic": textwrap.dedent("""\
            Photocatalytic water splitting for solar hydrogen production —
            catalyst materials (titanium dioxide, tantalum nitride,
            bismuth vanadate, carbon nitride, metal sulfide, perovskite
            oxynitride, 2D materials), Z-scheme / heterojunction
            architectures, cocatalysts (Pt, NiOx, CoOx), solar-to-hydrogen
            efficiency benchmarks, mechanism (hole scavenging, oxygen
            evolution half-reaction). Scope is semiconductor photocatalysis
            for H2 evolution — NOT photoelectrochemical (PEC) cells that
            require external bias, NOT photothermal H2 processes, NOT
            electrocatalytic HER without photocatalyst.
        """).strip(),
    },
    "hydride_sc": {
        "topic": textwrap.dedent("""\
            High-temperature superconductivity in hydrogen-rich compounds
            (hydrides / polyhydrides / superhydrides) under high pressure —
            covers H3S, LaH10, YH6/YH9, CaH6, CeH9/CeH10, and predicted /
            measured near-room-temperature superconducting hydrides;
            diamond-anvil cell synthesis, DFT + cluster-expansion structure
            search, Eliashberg electron-phonon coupling calculations,
            experimental Tc verification, and ongoing reproducibility
            debates. Scope is hydride superconductors at high pressure —
            NOT cuprate superconductors, NOT iron-pnictide, NOT
            conventional BCS low-Tc metals unless used as comparison.
        """).strip(),
    },
    "jwst_exo": {
        "topic": textwrap.dedent("""\
            Exoplanet atmosphere characterization using the James Webb
            Space Telescope — covers transmission spectroscopy, emission
            spectroscopy, phase curves, NIRISS / NIRSpec / MIRI
            instrument-specific reduction pipelines, atmospheric
            retrieval methods, detected molecular features (H2O, CO2,
            CH4, SO2, DMS claims), hot Jupiters, warm Neptunes, and
            rocky / temperate planet atmosphere observations. Scope is
            JWST-era exoplanet atmospheres — NOT Kepler / TESS
            detection papers, NOT radial velocity / astrometric discovery,
            NOT purely theoretical atmosphere modelling without JWST data.
        """).strip(),
    },
    "sei_battery": {
        "topic": textwrap.dedent("""\
            Solid electrolyte interphase (SEI) formation and stability in
            lithium-ion and lithium-metal batteries — covers SEI
            composition (inorganic / organic layers, LiF, Li2CO3, Li2O),
            formation mechanisms, cryo-electron microscopy and
            spectroscopy characterization, electrolyte additive effects
            (FEC, VC, LiNO3), artificial SEI design, high-voltage CEI
            counterpart, and SEI evolution during cycling. Scope is SEI
            chemistry and design in Li batteries — NOT general electrode
            material development without SEI focus, NOT sodium-ion or
            solid-state battery electrolyte bulk properties, NOT fuel-cell
            catalyst-layer interfaces.
        """).strip(),
    },
}


# Per-model YAML block (models: registry + search_model field)
MODEL_BLOCKS: dict[str, dict] = {
    "gemma": {
        "search_model": "gemma-4-31b",
        "models_block": textwrap.dedent("""\
            models:
              gemma-4-31b:
                base_url: "https://cola-lab--citeclaw-vllm-gemma-128k-serve.modal.run/v1"
                served_model_name: "google/gemma-4-31B-it"
                api_key_env: "CITECLAW_VLLM_API_KEY"
                reasoning_parser: "gemma4"
                max_model_len: 131072
            """).strip(),
    },
    "grok": {
        "search_model": "grok-4-fast-reasoning",
        "models_block": textwrap.dedent("""\
            models:
              grok-4-fast-reasoning:
                base_url: "https://api.x.ai/v1"
                served_model_name: "grok-4-1-fast-reasoning"
                api_key_env: "XAI_API_KEY"
                # grok-4-*-fast-reasoning reasons by default and rejects
                # the reasoning_effort kwarg; "none" skips the kwarg.
                reasoning_parser: "none"
            """).strip(),
    },
    "openai": {
        # OpenAI fallthrough — alias goes straight to OpenAIClient
        "search_model": "gpt-5.4-nano",
        "models_block": "",
    },
}


def build_config(scenario_id: str, model_key: str) -> str:
    if scenario_id not in SCENARIOS:
        raise KeyError(f"unknown scenario_id {scenario_id!r}")
    if model_key not in MODEL_BLOCKS:
        raise KeyError(f"unknown model_key {model_key!r}")
    sc = SCENARIOS[scenario_id]
    mb = MODEL_BLOCKS[model_key]
    data_dir = f"tests_v3/data/{scenario_id}_{model_key}"

    lines = [
        f"# {scenario_id} — {model_key}",
        'screening_model: "stub"',
        f'search_model: "{mb["search_model"]}"',
        "",
    ]
    if mb["models_block"]:
        lines.append(mb["models_block"])
        lines.append("")
    lines += [
        f'data_dir: "{data_dir}"',
        "max_papers_total: 1000",
        "",
        "topic_description: >",
    ]
    for t_line in sc["topic"].splitlines():
        lines.append("  " + t_line)
    lines += [
        "",
        "seed_papers:",
        '  - title: "placeholder (V3 does not show seeds to LLMs)"',
        "",
        "blocks:",
        "  noop_screener:",
        "    type: Sequential",
        "    layers: []",
        "",
        "pipeline:",
        "  - step: ExpandBySearchV3",
        "    agent:",
        "      max_subtopics: 5",
        "      max_iter: 5",
        "      max_papers_per_query: 10000",
        '      reasoning_effort: "medium"',
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: gen_config.py <scenario_id> <model_key>", file=sys.stderr)
        sys.exit(1)
    sid, mk = sys.argv[1], sys.argv[2]
    out = Path("tests_v3/configs") / f"{sid}_{mk}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_config(sid, mk))
    print(out)


if __name__ == "__main__":
    main()
