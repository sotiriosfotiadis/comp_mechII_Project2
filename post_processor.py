import numpy as np

class PostProcessor:
    def __init__(self, mesh, element, material):
        self.mesh = mesh
        self.element = element
        self.material = material

    def _element_dofs(self, node_ids, U):
        """Διάνυσμα μετατοπίσεων στοιχείου (12x1)."""
        u_e = np.zeros(12)
        for k, nid in enumerate(node_ids):
            idx = nid * 3
            u_e[3*k]     = U[idx]
            u_e[3*k + 1] = U[idx + 1]
            u_e[3*k + 2] = U[idx + 2]
        return u_e

    def stress_tensor_at_xy(self, element_idx, U, x, y):
        """
        Τάσεις [σx, σy, τxy] στην ΚΑΤΩΤΕΡΗ επιφάνεια, στο ΦΥΣΙΚΟ σημείο (x, y),
        από το στοιχείο `element_idx`.
        """
        node_ids = self.mesh.elements[element_idx]
        elem_nodes = self.mesh.nodes[node_ids]
        u_e = self._element_dofs(node_ids, U)

        center = elem_nodes.mean(axis=0)
        A = self.element.construct_A_matrix(elem_nodes - center)
        A_inv = np.linalg.inv(A)
        B = self.element.get_B_matrix_poly(x - center[0], y - center[1], A_inv)

        kappa = B @ u_e
        M = self.material.D @ kappa               # ροπές ανά μονάδα μήκους
        sigma = (6.0 / self.material.t**2) * M    # τάση ακραίας ίνας
        return sigma

    def calculate_element_stresses(self, element_idx, U, xi, eta):
        """Τάσεις στο φυσικό στοιχείο, από τις φυσικές συντεταγμένες (ξ, η)."""
        node_ids = self.mesh.elements[element_idx]
        elem_nodes = self.mesh.nodes[node_ids]
        xi_n = [-1, 1, 1, -1]
        eta_n = [-1, -1, 1, 1]
        x_p = 0.0
        y_p = 0.0
        for i in range(4):
            N_i = 0.25 * (1 + xi * xi_n[i]) * (1 + eta * eta_n[i])
            x_p += N_i * elem_nodes[i, 0]
            y_p += N_i * elem_nodes[i, 1]
        return self.stress_tensor_at_xy(element_idx, U, x_p, y_p)

    def principal_stress_at_point(self, U, x, y):
        """
        Τάσεις ΑΚΡΙΒΩΣ στο σημείο (x, y) — π.χ. στην τομή των διαγωνίων.
        Γίνεται μέσος όρος (nodal averaging) των στοιχείων που περιέχουν τον
        πλησιέστερο κόμβο. Επιστρέφει: (μέσο tensor, σ1, σ2).
        """
        target = np.array([x, y])
        dist2 = np.sum((self.mesh.nodes - target)**2, axis=1)
        node = int(np.argmin(dist2))

        tensors = []
        for ei, conn in enumerate(self.mesh.elements):
            if node in conn:
                tensors.append(self.stress_tensor_at_xy(ei, U, x, y))

        sigma = np.mean(tensors, axis=0)
        sx, sy, txy = sigma
        center = (sx + sy) / 2.0
        radius = np.sqrt(((sx - sy) / 2.0)**2 + txy**2)
        sigma_1 = center + radius
        sigma_2 = center - radius
        return sigma, sigma_1, sigma_2

    def export_displacements(self, U):
        return U.reshape((-1, 3))