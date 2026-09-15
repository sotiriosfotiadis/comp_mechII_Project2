import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend χωρίς παράθυρο (αποθήκευση σε αρχείο)
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from pre_processor import Mesh, Material, BoundaryConditions, generate_rhombus_mesh
from solver import KirchhoffPlateElement, Assembler, Solver


# --------------------------------------------------------------------------
#  Βοηθητικές συναρτήσεις
# --------------------------------------------------------------------------
def _boundary_edges(elements):
    """Επιστρέφει τις ακμές που ανήκουν σε ΕΝΑ μόνο στοιχείο (περίμετρος)."""
    counts, order = {}, {}
    for conn in elements:
        n = len(conn)
        for k in range(n):
            a, b = int(conn[k]), int(conn[(k + 1) % n])
            key = frozenset((a, b))
            counts[key] = counts.get(key, 0) + 1
            order.setdefault(key, (a, b))
    return [order[key] for key, c in counts.items() if c == 1]


def solve_plate(L=1.0, angle_deg=30, n_el=16, E=210e9, nu=0.3, t=0.01, q=-700.0):
    """Επιλύει το πρόβλημα και επιστρέφει (nodes, elements, w_nodes)."""
    nodes, elements = generate_rhombus_mesh(L=L, angle_deg=angle_deg, n_el=n_el)
    mesh = Mesh(nodes, elements)
    mat = Material(E=E, nu=nu, t=t)
    bc = BoundaryConditions()
    bc.set_pinned_edges(n_el=n_el)
    element = KirchhoffPlateElement(mesh, mat)
    asm = Assembler(mesh, element)
    K = asm.assemble_stiffness()
    F = asm.assemble_forces(q=q)
    K, F = bc.apply(K, F)
    U = Solver().solve(K, F)
    w_nodes = U.reshape((-1, 3))[:, 0]
    return nodes, elements, w_nodes


def _plate_corners(nodes, n_el):
    """Δείκτες κόμβων των 4 κορυφών του ρόμβου (με τη σειρά για κλειστό πολύγωνο)."""
    n_side = n_el + 1
    return [0, n_el, n_side * n_side - 1, n_el * n_side]


def _draw_outline_and_center(ax, nodes, elements, L, angle_deg):
    """Περίγραμμα πλάκας + κόκκινος σταυρός στην τομή των διαγωνίων (κέντρο)."""
    for a, b in _boundary_edges(elements):
        ax.plot([nodes[a, 0], nodes[b, 0]], [nodes[a, 1], nodes[b, 1]],
                color="0.15", lw=1.4)
    cx = L * (1 + np.cos(np.radians(angle_deg))) / 2
    cy = L * np.sin(np.radians(angle_deg)) / 2
    ax.plot(cx, cy, "r+", ms=11, mew=1.8)


# --------------------------------------------------------------------------
#  1) Αρχική κατάσταση — μηδενική βύθιση (w = 0)
# --------------------------------------------------------------------------
def export_initial_contour_2d(L=1.0, angle_deg=30, n_el=32, t=0.01,
                              E=210e9, nu=0.3, q=-700.0,
                              filename="plate_initial_contour.png", dpi=160):
    
    # Λύνουμε μόνο για να πάρουμε την ίδια κλίμακα χρώματος με το παραμορφωμένο
    nodes, elements, w = solve_plate(L, angle_deg, n_el, E, nu, t, q)
    w_mm = w * 1e3
    vmin, vmax = w_mm.min(), max(w_mm.max(), 0.0)

    cmap = cm.viridis
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Ομοιόμορφη πλήρωση της πλάκας στην τιμή w = 0
    corners = _plate_corners(nodes, n_el)
    ax.fill([nodes[c, 0] for c in corners], [nodes[c, 1] for c in corners],
            color=cmap(norm(0.0)))

    _draw_outline_and_center(ax, nodes, elements, L, angle_deg)

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Αρχική κατάσταση της πλάκας — μηδενική βύθιση (w = 0)")

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(w_mm)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.9, pad=0.03)
    cbar.set_label("κατακόρυφη μετατόπιση w (mm)")

    fig.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return filename


# --------------------------------------------------------------------------
#  2) Παραμορφωμένη κατάσταση — χάρτης ισοϋψών της βύθισης
# --------------------------------------------------------------------------
def export_deflection_contour_2d(L=1.0, angle_deg=30, n_el=32, t=0.01,
                                 E=210e9, nu=0.3, q=-700.0, n_levels=12,
                                 filename="plate_deflection_contour.png", dpi=160):
    
    nodes, elements, w = solve_plate(L, angle_deg, n_el, E, nu, t, q)

    n_side = n_el + 1
    X = nodes[:, 0].reshape(n_side, n_side)
    Y = nodes[:, 1].reshape(n_side, n_side)
    W = w.reshape(n_side, n_side) * 1e3  # mm

    levels = np.linspace(W.min(), W.max(), n_levels + 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    cf = ax.contourf(X, Y, W, levels=levels, cmap="viridis")
    cl = ax.contour(X, Y, W, levels=levels, colors="k", linewidths=0.5, alpha=0.6)
    ax.clabel(cl, inline=True, fontsize=7, fmt="%.3f")

    _draw_outline_and_center(ax, nodes, elements, L, angle_deg)

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Χάρτης ισοϋψών βύθισης w — {n_el}×{n_el} στοιχεία")

    cbar = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.03)
    cbar.set_label("κατακόρυφη μετατόπιση w (mm)")

    fig.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return filename


if __name__ == "__main__":
    f1 = export_initial_contour_2d(L=1.0, angle_deg=30, n_el=32, t=0.01)
    print(f"Αρχική κατάσταση (w=0)   -> {f1}")
    f2 = export_deflection_contour_2d(L=1.0, angle_deg=30, n_el=32, t=0.01)
    print(f"Χάρτης ισοϋψών βύθισης    -> {f2}")
