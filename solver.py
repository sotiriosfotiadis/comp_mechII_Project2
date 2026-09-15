import numpy as np

class KirchhoffPlateElement:
    def __init__(self, mesh, material):
        self.mesh = mesh
        self.mat = material

    def get_polynomial_terms(self, x, y):
        """
        Επιστρέφει τους όρους του πολυωνύμου και τις παραγώγους του.
        w = [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3, x^3y, xy^3] * a
        (μη-συμβατό στοιχείο πλάκας 12 β.ε., τύπου ACM)
        """
        P = np.array([1, x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**3*y, x*y**3])
        P_x = np.array([0, 1, 0, 2*x, y, 0, 3*x**2, 2*x*y, y**2, 0, 3*x**2*y, y**3])
        P_y = np.array([0, 0, 1, 0, x, 2*y, 0, x**2, 2*x*y, 3*y**2, x**3, 3*x*y**2])
        P_xx = np.array([0, 0, 0, 2, 0, 0, 6*x, 2*y, 0, 0, 6*x*y, 0])       # w,xx
        P_yy = np.array([0, 0, 0, 0, 0, 2, 0, 0, 2*x, 6*y, 0, 6*x*y])       # w,yy
        P_xy = np.array([0, 0, 0, 0, 1, 0, 0, 2*x, 2*y, 0, 3*x**2, 3*y**2]) # w,xy
        return P, P_x, P_y, P_xx, P_yy, P_xy

    def construct_A_matrix(self, nodes):
        """
        Κατασκευάζει τον πίνακα A (12x12) αντικαθιστώντας τις συντεταγμένες
        των 4 κόμβων στο πολυώνυμο (ΤΟΠΙΚΕΣ συντεταγμένες ως προς το κέντρο).
        """
        A = np.zeros((12, 12))
        for i in range(4):
            xi, yi = nodes[i, 0], nodes[i, 1]
            P, P_x, P_y, _, _, _ = self.get_polynomial_terms(xi, yi)
            A[3*i, :] = P          # w
            A[3*i+1, :] = -P_y     # theta_x = -w,y
            A[3*i+2, :] = P_x      # theta_y = +w,x
        return A

    def get_B_matrix_poly(self, x, y, A_inv):
        """Curvatures = [w,xx, w,yy, 2w,xy]^T  ->  B = L * A_inv."""
        _, _, _, P_xx, P_yy, P_xy = self.get_polynomial_terms(x, y)
        L = np.zeros((3, 12))
        L[0, :] = P_xx
        L[1, :] = P_yy
        L[2, :] = 2.0 * P_xy
        return L @ A_inv

    def get_N_matrix_poly(self, x, y, A_inv):
        """Συναρτήσεις σχήματος N = P * A_inv (τοπικές x, y)."""
        P, _, _, _, _, _ = self.get_polynomial_terms(x, y)
        return P @ A_inv

    # 3x3 Gauss: ακριβής και για τους όρους στρέψης (x^2, y^2)
    _GP = [-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]
    _GW = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
    _XI_CORNERS = [-1, 1, 1, -1]
    _ETA_CORNERS = [-1, -1, 1, 1]

    def _local_setup(self, elem_nodes):
        """Κέντρο στοιχείου + A_inv σε τοπικές συντεταγμένες."""
        center = elem_nodes.mean(axis=0)
        A = self.construct_A_matrix(elem_nodes - center)
        A_inv = np.linalg.inv(A)
        return center, A_inv

    def ke(self, elem_nodes):
        center, A_inv = self._local_setup(elem_nodes)
        ke = np.zeros((12, 12))
        D = self.mat.D
        for a, r in enumerate(self._GP):
            for b, s in enumerate(self._GP):
                x_g = 0.0; y_g = 0.0
                dx_dxi = 0.0; dx_deta = 0.0; dy_dxi = 0.0; dy_deta = 0.0
                for i in range(4):
                    N_i = 0.25 * (1 + r * self._XI_CORNERS[i]) * (1 + s * self._ETA_CORNERS[i])
                    dN_dxi = 0.25 * self._XI_CORNERS[i] * (1 + s * self._ETA_CORNERS[i])
                    dN_deta = 0.25 * self._ETA_CORNERS[i] * (1 + r * self._XI_CORNERS[i])
                    x_g += N_i * elem_nodes[i, 0]
                    y_g += N_i * elem_nodes[i, 1]
                    dx_dxi += dN_dxi * elem_nodes[i, 0]
                    dy_dxi += dN_dxi * elem_nodes[i, 1]
                    dx_deta += dN_deta * elem_nodes[i, 0]
                    dy_deta += dN_deta * elem_nodes[i, 1]
                det_J = dx_dxi * dy_deta - dx_deta * dy_dxi
                B = self.get_B_matrix_poly(x_g - center[0], y_g - center[1], A_inv)
                ke += (B.T @ D @ B) * det_J * self._GW[a] * self._GW[b]
        return ke

    def calculate_load_vector(self, elem_nodes, q):
        center, A_inv = self._local_setup(elem_nodes)
        fe = np.zeros(12)
        for a, r in enumerate(self._GP):
            for b, s in enumerate(self._GP):
                x_g = 0.0; y_g = 0.0
                dx_dxi = 0.0; dx_deta = 0.0; dy_dxi = 0.0; dy_deta = 0.0
                for i in range(4):
                    N_i = 0.25 * (1 + r * self._XI_CORNERS[i]) * (1 + s * self._ETA_CORNERS[i])
                    dN_dxi = 0.25 * self._XI_CORNERS[i] * (1 + s * self._ETA_CORNERS[i])
                    dN_deta = 0.25 * self._ETA_CORNERS[i] * (1 + r * self._XI_CORNERS[i])
                    x_g += N_i * elem_nodes[i, 0]
                    y_g += N_i * elem_nodes[i, 1]
                    dx_dxi += dN_dxi * elem_nodes[i, 0]
                    dy_dxi += dN_dxi * elem_nodes[i, 1]
                    dx_deta += dN_deta * elem_nodes[i, 0]
                    dy_deta += dN_deta * elem_nodes[i, 1]
                det_J = dx_dxi * dy_deta - dx_deta * dy_dxi
                N = self.get_N_matrix_poly(x_g - center[0], y_g - center[1], A_inv)
                fe += N * q * det_J * self._GW[a] * self._GW[b]
        return fe


class Assembler:
    def __init__(self, mesh, element):
        self.mesh = mesh
        self.element = element

    def _global_dofs(self, conn):
        g_dofs = []
        for nid in conn:
            start = nid * 3
            g_dofs.extend([start, start + 1, start + 2])
        return g_dofs

    def assemble_stiffness(self):
        ndof = self.mesh.ndof
        K = np.zeros((ndof, ndof))
        for conn in self.mesh.elements:
            elem_coords = self.mesh.nodes[conn]
            ke = self.element.ke(elem_coords)
            g_dofs = self._global_dofs(conn)
            for i in range(12):
                for j in range(12):
                    K[g_dofs[i], g_dofs[j]] += ke[i, j]
        return K

    def assemble_forces(self, q):
        ndof = self.mesh.ndof
        F = np.zeros(ndof)
        for conn in self.mesh.elements:
            elem_coords = self.mesh.nodes[conn]
            fe = self.element.calculate_load_vector(elem_coords, q)
            g_dofs = self._global_dofs(conn)
            for i in range(12):
                F[g_dofs[i]] += fe[i]
        return F


class Solver:
    def solve(self, K, F):
        return np.linalg.solve(K, F)
