import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import spacy
from spacy.matcher import PhraseMatcher
from pint import UnitRegistry

from .pde_spec import PDESpec, GeometrySpec, MaterialSpec, MeshSpec, BCSpeс  # keep your class name as-is

# =======================
# NLP + Units setup
# =======================

_NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])

_ureg = UnitRegistry()
_Q_ = _ureg.Quantity

_ureg.define("um = micrometer = micrometre")
_ureg.define("nm = nanometer = nanometre")
_ureg.define("kN = kilonewton")
_ureg.define("K = kelvin")

# =======================
# Domain/material vocab
# =======================

DOMAIN_KEYWORDS = {
    "heat_transfer": ["heat", "thermal", "temperature", "conduction", "diffusion"],
    "solid_mechanics": ["stress", "strain", "deformation", "elastic", "cantilever", "beam", "young", "poisson"],
    "fluid": ["fluid", "navier", "stokes", "flow", "velocity", "pressure"],
}

MATERIALS = [
    "copper",
    "aluminum",
    "aluminium",
    "silicon",
    "steel",
    "gold",
    "silver",
]

GEOM_KEYWORDS = {
    "thin_film": ["thin film", "film", "thin layer", "membrane"],
    "beam": ["beam", "cantilever"],
    "box": ["block", "box", "slab"],
    "sphere": ["sphere", "ball"],
}

_material_matcher = PhraseMatcher(_NLP.vocab, attr="LOWER")
_material_matcher.add("MATERIAL", [_NLP.make_doc(m) for m in MATERIALS])

_geom_matcher = PhraseMatcher(_NLP.vocab, attr="LOWER")
for k, phrases in GEOM_KEYWORDS.items():
    _geom_matcher.add(f"GEOM_{k.upper()}", [_NLP.make_doc(p) for p in phrases])

# =======================
# Data model for extraction
# =======================

@dataclass
class PromptEntities:
    domain: str
    material: Optional[str] = None

    thickness_m: Optional[float] = None
    lengths_m: List[float] = field(default_factory=list)
    geom_hint: Optional[str] = None  # thin_film, beam, box, sphere

    temperatures_K: List[float] = field(default_factory=list)
    force_N: Optional[float] = None

    tokens: List[str] = field(default_factory=list)
    matches: Dict[str, Any] = field(default_factory=dict)

# =======================
# Unit extraction helpers
# =======================

_QUANTITY_RE = re.compile(
    r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([a-zA-Zµ]+)\b"
)

def _parse_quantities(text: str):
    out = []
    for m in _QUANTITY_RE.finditer(text):
        val_str, unit_str = m.group(1), m.group(2)
        raw = m.group(0)
        unit_norm = unit_str.replace("µ", "u")

        try:
            q = _Q_(float(val_str), unit_norm)
        except Exception:
            continue

        try:
            if q.check("[length]"):
                out.append((q.to("m").magnitude, "length", raw, unit_str))
            elif q.check("[force]"):
                out.append((q.to("N").magnitude, "force", raw, unit_str))
            elif str(q.units).lower() in ("kelvin", "k"):
                out.append((q.to("K").magnitude, "temperature", raw, unit_str))
        except Exception:
            continue

    return out

# =======================
# Domain inference
# =======================

def _infer_domain(prompt: str) -> str:
    doc = _NLP(prompt.lower())
    text = " ".join([t.text for t in doc])
    for domain, keys in DOMAIN_KEYWORDS.items():
        if any(k in text for k in keys):
            return domain
    return "other"

def _infer_material(doc) -> Optional[str]:
    matches = _material_matcher(doc)
    if not matches:
        return None
    _, start, end = matches[0]
    return doc[start:end].text.lower()

def _infer_geom_hint(doc) -> Optional[str]:
    matches = _geom_matcher(doc)
    if not matches:
        return None

    # Sphere must beat beam when both appear (e.g., "cantilever sphere")
    priorities = {
        "GEOM_SPHERE": 4,
        "GEOM_THIN_FILM": 3,
        "GEOM_BEAM": 2,
        "GEOM_BOX": 1,
    }

    best = None
    best_p = -1
    for match_id, start, end in matches:
        label = doc.vocab.strings[match_id]
        p = priorities.get(label, 0)
        if p > best_p:
            best_p = p
            best = label

    if best == "GEOM_THIN_FILM":
        return "thin_film"
    if best == "GEOM_BEAM":
        return "beam"
    if best == "GEOM_BOX":
        return "box"
    if best == "GEOM_SPHERE":
        return "sphere"
    return None

# =======================
# Extraction
# =======================

def extract_entities(prompt: str, domain_hint: Optional[str] = None) -> PromptEntities:
    doc = _NLP(prompt)

    domain = domain_hint or _infer_domain(prompt)
    ent = PromptEntities(domain=domain)
    ent.tokens = [t.text for t in doc]

    ent.material = _infer_material(doc)
    ent.geom_hint = _infer_geom_hint(doc)

    quantities = _parse_quantities(prompt)
    lengths = [v for (v, qtype, raw, unit) in quantities if qtype == "length"]
    temps   = [v for (v, qtype, raw, unit) in quantities if qtype == "temperature"]
    forces  = [v for (v, qtype, raw, unit) in quantities if qtype == "force"]

    ent.lengths_m = sorted(lengths)
    ent.temperatures_K = temps

    # keep thickness heuristic (used for thin-film heat, etc.)
    if lengths:
        ent.thickness_m = min(lengths)

    if forces:
        ent.force_N = forces[0]

    ent.matches = {
        "quantities": quantities,
        "material": ent.material,
        "geom_hint": ent.geom_hint,
    }
    print(ent.matches["quantities"])
    return ent

# =======================
# Template builders
# =======================

def _heat_from_entities(ent: PromptEntities) -> PDESpec:
    extracted: Dict[str, Any] = {"entities": ent.matches}
    warnings: List[str] = []
    questions: List[str] = []

    material_name = ent.material
    mat_props: Dict[str, Any] = {}

    if material_name == "copper":
        mat_props["k"] = 400.0
    elif material_name in ("aluminum", "aluminium"):
        mat_props["k"] = 205.0
    elif material_name == "silicon":
        mat_props["k"] = 148.0
    elif material_name:
        warnings.append(f"Material '{material_name}' recognized but no default properties set; using k=1.0 W/mK.")
        mat_props["k"] = 1.0
    else:
        questions.append("What material is it (e.g., copper, silicon, steel)?")

    geom_type = ent.geom_hint or "box"

    # ---- Heat geometry dims: ONLY {R} for sphere, or {Lx,Ly,Lz} for box/thin_film ----
    if geom_type == "sphere":
        # If prompt said "radius", the number is usually the only length → use it as R
        if ent.lengths_m:
            R = float(ent.lengths_m[0])  # if only one length, this is it
        else:
            R = 1.0
            questions.append("What is the sphere radius (e.g., 2 m)?")
        dims: Dict[str, float] = {"R": R}
        warnings.append(f"Assuming sphere geometry with radius R={R} m (override recommended).")
    else:
        dims = {"Lx": 1.0, "Ly": 1.0, "Lz": 0.0}  # 2D default

        Ls = sorted(ent.lengths_m) if ent.lengths_m else []
        if len(Ls) >= 3:
            dims["Lz"] = float(Ls[0])
            dims["Ly"] = float(Ls[-2])
            dims["Lx"] = float(Ls[-1])
            extracted["mapped_lengths_m"] = Ls
            extracted["thickness_m"] = float(Ls[0])
            warnings.append("Mapped 3+ length values to (Lx, Ly, Lz) = (largest, middle, smallest).")
        elif len(Ls) == 2:
            dims["Lz"] = float(Ls[0])
            dims["Lx"] = float(Ls[1])
            extracted["mapped_lengths_m"] = Ls
            extracted["thickness_m"] = float(Ls[0])
            warnings.append("Mapped 2 length values to (Lx, Lz) = (largest, smallest); using default Ly=1.0 m.")
        elif len(Ls) == 1:
            # If user said thin-film-ish words: treat as thickness; else ask
            p = " ".join(ent.tokens).lower() if ent.tokens else ""
            thickness_words = ["thickness", "thick", "thin", "film", "layer", "membrane"]
            if any(w in p for w in thickness_words):
                dims["Lz"] = float(Ls[0])
                extracted["thickness_m"] = float(Ls[0])
                warnings.append("Single length treated as thickness (based on prompt keywords). Using default Lx=Ly=1.0 m.")
            else:
                questions.append("You provided one length. Is it the thickness (Lz) or an in-plane dimension (Lx/Ly)?")
                warnings.append("Keeping default (Lx,Ly,Lz) until thickness/in-plane dimension is clarified.")
        else:
            warnings.append("No lengths found; using default box 1m x 1m (2D). Override recommended.")

    # BCs: Dirichlet left/right (your solver currently supports this)
    if len(ent.temperatures_K) >= 2:
        T_left, T_right = ent.temperatures_K[0], ent.temperatures_K[1]
        bc = {"type": "dirichlet_lr", "T_left": float(T_left), "T_right": float(T_right)}
    else:
        questions.append("What boundary temperatures should I use (e.g., 500 K on left and 300 K on right)?")
        bc = {"type": "dirichlet_lr", "T_left": None, "T_right": None}

    # Mesh defaults (sphere will use mesh.params['h'] in your gmsh heat solver; keep nx/ny/nz anyway)
    mesh = MeshSpec(resolution="auto", params={"nx": 40, "ny": 40, "nz": 2})
    if geom_type == "sphere":
        # a reasonable default: h ~ R/10 if user didn't specify
        if "h" not in mesh.params:
            mesh.params["h"] = float(dims["R"]) / 10.0

    return PDESpec(
        domain="heat_transfer",
        pde="steady_heat",
        geometry=GeometrySpec(type=geom_type, dims=dims),
        material=MaterialSpec(name=material_name, props=mat_props),
        bc=BCSpeс(kind=bc),
        mesh=mesh,
        outputs=["temperature"],
        units="SI",
        extracted=extracted,
        warnings=warnings,
        clarifying_questions=questions,
    )

def _solid_from_entities(ent: PromptEntities) -> PDESpec:
    extracted: Dict[str, Any] = {"entities": ent.matches}
    warnings: List[str] = []
    questions: List[str] = []

    if ent.force_N is None:
        questions.append("What load should be applied (e.g., 1000 N tip load)?")

    geom_type = ent.geom_hint or "beam"

    if geom_type == "sphere":
        if ent.lengths_m:
            R = float(ent.lengths_m[0])
        else:
            R = 1.0
            questions.append("What is the sphere radius (e.g., 2 m)?")
        dims = {"R": R}
        warnings.append(f"Assuming sphere geometry with radius R={R} m (override recommended).")
    else:
        dims = {"L": 0.1, "W": 0.01, "H": 0.01}
        warnings.append("Assuming beam geometry L=0.1m, W=0.01m, H=0.01m (override recommended).")

    material_name = ent.material
    mat_props: Dict[str, Any] = {}

    if material_name == "steel":
        mat_props.update({"E": 200e9, "nu": 0.30})
    elif material_name:
        warnings.append(f"Material '{material_name}' recognized but no default elastic properties set.")
        questions.append("Provide material properties (E, nu) or should I assume E=1e9 Pa, nu=0.30?")
        mat_props.update({"E": 1e9, "nu": 0.30})
    else:
        questions.append("What material is it (e.g., steel, aluminum)?")

    bc = {
        "type": "cantilever_tip_load",
        "fixed_end": "x=0",
        "load_end": "x=L",
        "F": float(ent.force_N) if ent.force_N is not None else None,
        "direction": "down",
    }

    mesh = MeshSpec(resolution="auto", params={"nx": 60, "ny": 10, "nz": 10})
    if geom_type == "sphere":
        if "h" not in mesh.params:
            mesh.params["h"] = float(dims["R"]) / 10.0

    return PDESpec(
        domain="solid_mechanics",
        pde="linear_elasticity",
        geometry=GeometrySpec(type=geom_type, dims=dims),
        material=MaterialSpec(name=material_name, props=mat_props),
        bc=BCSpeс(kind=bc),
        mesh=mesh,
        outputs=["displacement", "von_mises"],
        units="SI",
        extracted=extracted,
        warnings=warnings,
        clarifying_questions=questions,
    )

def _fluid_from_entities(ent: PromptEntities) -> PDESpec:
    extracted = {"entities": ent.matches}
    warnings: List[str] = []
    questions: List[str] = []

    Lx = getattr(ent, "Lx_m", None) or 1.0
    Ly = getattr(ent, "Ly_m", None) or 0.2
    Lz = getattr(ent, "Lz_m", None) or 0.2

    Uin = getattr(ent, "inlet_velocity_mps", None)
    if Uin is None:
        questions.append("What inlet velocity should I use (e.g., 0.2 m/s)?")
        Uin = 0.2
        warnings.append("No inlet velocity found; using default 0.2 m/s.")

    mu = getattr(ent, "mu_Pas", None)
    if mu is None:
        mu = 1e-3
        warnings.append("No viscosity provided; using mu=1e-3 Pa·s (water-like).")

    geom = GeometrySpec(type="box", dims={"Lx": float(Lx), "Ly": float(Ly), "Lz": float(Lz)})
    mat = MaterialSpec(name="fluid", props={"mu": float(mu)})

    bc = BCSpeс(kind={
        "type": "channel_inlet_outlet",
        "U_in": float(Uin),
        "p_out": 0.0
    })

    mesh = MeshSpec(resolution="auto", params={"nx": 40, "ny": 16, "nz": 16})

    return PDESpec(
        domain="fluid",
        pde="steady_stokes",
        geometry=geom,
        material=mat,
        bc=bc,
        mesh=mesh,
        outputs=["velocity_mag", "pressure"],
        units="SI",
        extracted=extracted,
        warnings=warnings,
        clarifying_questions=questions,
    )

# =======================
# Patch logic for reuse
# =======================

def _spec_from_parameters(parameters: dict) -> Optional[PDESpec]:
    if not isinstance(parameters, dict):
        return None
    if "domain" in parameters and "geometry" in parameters and "material" in parameters:
        return PDESpec.model_validate(parameters)
    return None

def _patch_from_entities(ent: PromptEntities) -> Dict[str, Any]:
    patch: Dict[str, Any] = {}

    if ent.material:
        patch.setdefault("material", {})["name"] = ent.material

    geom_hint = ent.geom_hint
    if geom_hint:
        patch.setdefault("geometry", {})["type"] = geom_hint

    dims_patch: Dict[str, Any] = {}

    if geom_hint == "sphere":
        # Use smallest/only length as radius
        if ent.lengths_m:
            dims_patch["R"] = float(ent.lengths_m[0])
        elif ent.thickness_m is not None:
            dims_patch["R"] = float(ent.thickness_m)
    else:
        if getattr(ent, "Lx_m", None) is not None:
            dims_patch["Lx"] = float(ent.Lx_m)
        if getattr(ent, "Ly_m", None) is not None:
            dims_patch["Ly"] = float(ent.Ly_m)
        if ent.thickness_m is not None:
            dims_patch["Lz"] = float(ent.thickness_m)

    if dims_patch:
        patch.setdefault("geometry", {}).setdefault("dims", {}).update(dims_patch)

    if ent.domain == "heat_transfer" and ent.temperatures_K:
        temps = ent.temperatures_K
        bc_patch: Dict[str, Any] = {}
        if len(temps) >= 1:
            bc_patch["T_left"] = float(temps[0])
        if len(temps) >= 2:
            bc_patch["T_right"] = float(temps[1])
        if bc_patch:
            patch.setdefault("bc", {}).update(bc_patch)

    return patch

# =======================
# Main entry point
# =======================

def parse_prompt_to_spec(
    prompt: str,
    domain_hint: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> PDESpec:
    ent = extract_entities(prompt, domain_hint=domain_hint)
    domain = ent.domain

    base_spec = _spec_from_parameters(parameters) if parameters else None
    if base_spec is not None and domain == base_spec.domain:
        patch = _patch_from_entities(ent)

        # material
        if "material" in patch and "name" in patch["material"]:
            base_spec.material.name = patch["material"]["name"]

        # geometry
        if "geometry" in patch:
            gtype = patch["geometry"].get("type", None)
            if gtype is not None:
                base_spec.geometry.type = gtype

            gd = patch["geometry"].get("dims", {})
            if gd:
                base_spec.geometry.dims.update(gd)

        # bc
        if "bc" in patch:
            base_spec.bc.kind.update(patch["bc"])

        if domain_hint and domain_hint != base_spec.domain:
            base_spec.domain = domain_hint

        base_spec.extracted = base_spec.extracted or {}
        base_spec.extracted["entities"] = ent.matches
        base_spec.warnings.append("Reused previous setup; applied updates from follow-up prompt.")

        # ---- Domain-specific safety belt ----
        if base_spec.domain == "heat_transfer" and base_spec.geometry.type == "beam":
            base_spec.geometry.type = "box"
            base_spec.warnings.append("Geometry 'beam' not supported for heat; treating as 'box'.")

        return base_spec

    # Fresh spec
    if domain == "heat_transfer":
        spec = _heat_from_entities(ent)
    elif domain == "solid_mechanics":
        spec = _solid_from_entities(ent)
    elif domain == "fluid":
        spec = _fluid_from_entities(ent)
        spec.domain = "fluid"
        spec.pde = "steady_stokes"
    else:
        inferred = _infer_domain(prompt)
        if inferred == "solid_mechanics":
            spec = _solid_from_entities(ent)
        elif inferred == "fluid":
            spec = _fluid_from_entities(ent)
            spec.domain = "fluid"
            spec.pde = "steady_stokes"
        else:
            spec = _heat_from_entities(ent)
            spec.domain = "other"
            spec.warnings.append("Domain unclear; defaulting to heat template. Please specify domain.")

    # Overrides dict (non-full-spec) merge
    if parameters and base_spec is None:
        if "geometry" in parameters:
            spec.geometry.dims.update(parameters["geometry"].get("dims", {}))
            if "type" in parameters["geometry"]:
                spec.geometry.type = parameters["geometry"]["type"]

        if "material" in parameters:
            if "name" in parameters["material"]:
                spec.material.name = parameters["material"]["name"]
            spec.material.props.update(parameters["material"].get("props", {}))

        if "bc" in parameters:
            spec.bc.kind.update(parameters["bc"])

        if "mesh" in parameters and "params" in parameters["mesh"]:
            spec.mesh.params.update(parameters["mesh"]["params"])

        if "outputs" in parameters and isinstance(parameters["outputs"], list):
            spec.outputs = parameters["outputs"]

    # ---- Domain-specific safety belt ----
    if spec.domain == "heat_transfer" and spec.geometry.type == "beam":
        spec.geometry.type = "box"
        spec.warnings.append("Geometry 'beam' not supported for heat; treating as 'box'.")

    return spec
