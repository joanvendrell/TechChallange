# PhysicsSim (Natural-Language Physics Simulations)

Build a web-based interface that allows users to run physics simulations (including but not limited to heat transfer and solid mechanics) directly from natural language prompts. For example, a user might type: **“Simulate the heat transfer through a thin 30 nm conductor.”**

## Approach

Adopt an **agentic architecture** (e.g., master-agent loops and MCPs) to orchestrate the simulator.

### Your system should

1. **Provide “zero-barrier” simulations**  
   The app should be hosted with everything preinstalled, so all a user needs to do is type in a prompt.

2. **Interpret the prompt**  
   Parse the natural language input and configure the appropriate PDE system.

3. **Set up and solve the model**  
   Use finite element simulations powered by **FEniCS / DOLFINx** on a backend server.

4. **Deliver results interactively**  
   Render the simulation results in the frontend as an interactive **3D visualization**, allowing users to zoom, rotate, and explore computed fields.

5. **Adaptability**  
   Easily extend to new PDEs, boundary conditions, materials, and geometries by adding:
   - parsing rules / entities
   - new solver modules
   - new frontend field renderers

---

## Examples

### 1) Heat Transfer

**Prompt**
> Simulate steady heat transfer through an aluminum plate 5 m by 5 m by 1 m. Keep the left boundary at 500 K and apply convection on the right with ambient temperature 300 K.

![Heat transfer example](images/heat.png)

---

### 2) Solid Mechanics

**Prompt**
> Simulate the linear elastic deformation of a steel sphere 2 m radius, fixed at x=0 and subjected to a 1000 N downward tip load.

![Solid mechanics example](images/solid.png)

---

### 3) Fluids

**Prompt**
> Simulate steady Stokes flow in a 1 m by 0.2 m by 0.2 m channel with inlet velocity 0.3 m/s and outlet pressure 0 Pa.

![Fluids example](imgaes/fluids.png)

---

## High-Level Architecture

- **Frontend**
  - Prompt input (natural language)
  - Interactive 3D viewer (rotate/zoom/pan)
  - Field selection (e.g., temperature / von Mises / velocity magnitude)

- **Backend**
  - Prompt parser → `PDESpec` (domain, PDE, geometry, material, mesh, BCs, outputs)
  - Solver execution (FEniCS / DOLFINx)
  - Artifact export (e.g., `.vtp` / VTK formats)
  - Results API (metadata + artifact URLs)

- **Orchestrator (Agentic Loop)**
  - Routes prompt to the correct pipeline (heat/solid/fluid/…)
  - Requests missing info when needed (clarifying questions)
  - Applies safe defaults when possible
  - Produces structured outputs (fields, units, ranges, artifacts)

---

## Getting Started (Typical)

> Update these sections to match your repo layout and scripts.

### Backend

```bash
cd backend
conda env create -f environment.yml
conda activate physics-sim
uvicorn app.main:app --reload
