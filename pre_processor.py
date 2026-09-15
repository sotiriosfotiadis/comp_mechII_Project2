import numpy as np

def generate_rhombus_mesh(L, angle_deg, n_el):
    """
    Δομημένο πλέγμα ρομβοειδούς πλάκας πλευράς L και οξείας γωνίας `angle_deg`.
    Επιστρέφει (nodes, elements) με στοιχεία 4 κόμβων (Q4).
    """
    theta = np.radians(angle_deg)

    nodes = []
    elements = []

    n_nodes_side = n_el + 1

    for j in range(n_nodes_side):
        for i in range(n_nodes_side):
            x = (i / n_el) * L + (j / n_el) * L * np.cos(theta)
            y = (j / n_el) * L * np.sin(theta)
            nodes.append((x, y))

    for j in range(n_el):
        for i in range(n_el):
            n1 = j * n_nodes_side + i
            n2 = n1 + 1
            n3 = n2 + n_nodes_side
            n4 = n1 + n_nodes_side
            elements.append((n1, n2, n3, n4))

    return np.array(nodes), np.array(elements)


class Mesh:
    def __init__(self, nodes, elements):
        self.nodes = np.asarray(nodes)
        self.elements = np.asarray(elements, dtype=int)
        self.ndof_per_node = 3
        self.ndof = self.nodes.shape[0] * self.ndof_per_node


class Material:
    def __init__(self, E, nu, t):
        self.E = E
        self.nu = nu
        self.t = t
        self.D_factor = E * t**3 / (12 * (1 - nu**2))
        self.D = self.D_factor * np.array([
            [1,  nu, 0],
            [nu, 1,  0],
            [0,  0,  (1 - nu) / 2]
        ])


class BoundaryConditions:
    def __init__(self):
        self.fixed = []
        self.loads = {}

    def set_pinned_edges(self, n_el):
        """Απλές αρθρώσεις σε όλη την περίμετρο: δεσμεύεται ΜΟΝΟ το w (ελεύθερες περιστροφές)."""
        n_nodes_side = n_el + 1
        for j in range(n_nodes_side):
            for i in range(n_nodes_side):
                nid = j * n_nodes_side + i
                if i == 0 or i == n_el or j == 0 or j == n_el:
                    self.fixed.append((nid, 0))  # fix deflection w only

    def apply(self, K, F, ndof_per_node=3):
        for dof, val in self.loads.items():
            F[dof] += val
        for (nid, d) in self.fixed:
            gdof = nid * ndof_per_node + d
            K[gdof, :] = 0.0
            K[:, gdof] = 0.0
            K[gdof, gdof] = 1.0
            F[gdof] = 0.0
        return K, F

