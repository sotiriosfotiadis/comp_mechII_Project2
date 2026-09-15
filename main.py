from pre_processor import Mesh, Material, BoundaryConditions, generate_rhombus_mesh
from solver import KirchhoffPlateElement, Assembler, Solver
from post_processor import PostProcessor
import numpy as np

if __name__ == "__main__":
    L = 1.0
    angle_deg = 30
    ref_val = 0.802  # Τιμή αναφοράς μέγιστης κύριας τάσης (MPa) - NAFEMS

    # Παράμετροι προβλήματος
    E = 210e9      # Pa
    nu = 0.3
    t = 0.01       # m
    q_load = -700.0  # Pa  (q = -0.7 kPa)

    # Πλέγματα ελέγχου σύγκλισης (4x4, 8x8, 16x16, 32x32)
    mesh_sizes = [4, 8, 16, 32 ]

    print("\n" + "=" * 78)
    print("ΑΝΑΛΥΣΗ ΣΥΓΚΛΙΣΗΣ ΡΟΜΒΟΕΙΔΟΥΣ ΠΛΑΚΑΣ KIRCHHOFF (γωνία 30°)")
    print("=" * 78)
    print(f"{'n_el':<8}{'n_nodes':<10}{'|w_max| (m)':<18}{'σ1 (MPa)':<14}{'Error (%)':<10}")
    print("-" * 78)

    for n_el in mesh_sizes:
        nodes, elements = generate_rhombus_mesh(L=L, angle_deg=angle_deg, n_el=n_el)
        mesh = Mesh(nodes, elements)
        n_nodes = mesh.nodes.shape[0]

        mat = Material(E=E, nu=nu, t=t)

        bc = BoundaryConditions()
        bc.set_pinned_edges(n_el=n_el)

        element = KirchhoffPlateElement(mesh, mat)
        asm = Assembler(mesh, element)
        K = asm.assemble_stiffness()
        F = asm.assemble_forces(q=q_load)
        K, F = bc.apply(K, F)

        U = Solver().solve(K, F)

        post = PostProcessor(mesh, element, mat)
        d = post.export_displacements(U)
        max_w = abs(np.min(d[:, 0]))  # μέγιστη (κατά μέτρο) κατακόρυφη μετατόπιση

        # Τάσεις στην κάτω επιφάνεια, ΑΚΡΙΒΩΣ στην τομή των διαγωνίων (κέντρο ρόμβου)
        center_x = L * (1 + np.cos(np.radians(angle_deg))) / 2
        center_y = L * np.sin(np.radians(angle_deg)) / 2
        _, sigma_1, _ = post.principal_stress_at_point(U, center_x, center_y)
        sigma_1_mpa = sigma_1 / 1e6

        error_perc = abs((sigma_1_mpa - ref_val) / ref_val) * 100

        print(f"{n_el:<8}{n_nodes:<10}{max_w:<18.6e}{sigma_1_mpa:<14.4f}{error_perc:<10.2f}")

    print("=" * 78)
    print(f"Τιμή αναφοράς μέγιστης κύριας τάσης: σ* = {ref_val} MPa (NAFEMS)")
    print("=" * 78 + "\n")

