import pandas
import numpy as np
from pylmesh import Mesh
from trimesh import Trimesh
from neurom.core import Morphology
from scipy.spatial import KDTree
from conntility import subcellular
from scipy.stats import norm

Y_OPTIONS = ["f_factor", "f_factor_neck", "f_factor_head", "head_neck_ratio", "spine_density"]
Y_MINIMA = {
    "f_factor": 1.0,
    "f_factor_neck": 1.0,
    "f_factor_head": 1.0,
    "head_neck_ratio": 0.0,
    "spine_density": 0.0
}

def _simplify(vertices, faces, factor):
    tmesh = Trimesh(vertices=vertices, faces=faces)
    tmesh = tmesh.simplify_quadric_decimation(percent=factor)
    return np.array(tmesh.vertices), np.array(tmesh.faces)

def mesh_vertices_and_faces(neuron_mesh: Mesh, simplify=None):
    vertices_flat = neuron_mesh.get_vertices_array()
    vertices = np.array(vertices_flat).reshape(-1, 3) * 1E-3

    faces_flat = neuron_mesh.get_faces_array()
    faces = np.array(faces_flat).reshape(-1, 3)
    if simplify is not None:
        return _simplify(vertices, faces, simplify)
    return vertices, faces

def morph_segments_dataframe(morph: Morphology, ):
    pt_df = []
    for sec_ in morph.sections:
        seg_centers = 0.5 * (sec_.points[:-1, :-1] + sec_.points[1:, :-1])
        seg_l = np.linalg.norm((sec_.points[:-1, :-1] - sec_.points[1:, :-1]), axis=1)
        seg_r = 0.5 * (sec_.points[:-1, -1] + sec_.points[1:, -1])
        df_ = pandas.DataFrame(seg_centers, columns=["x", "y", "z"])
        df_["afferent_section_id"] = sec_.id + 1
        df_["afferent_segment_id"] = np.arange(len(df_))
        df_["afferent_segment_offset"] = 0.5 * seg_l
        df_["cone_area"] = np.pi * seg_l * (seg_r ** 2)
        df_["length"] = seg_l
        df_["radius"] = seg_r
        pt_df.append(df_)

    pt_df = pandas.concat(pt_df, axis=0).reset_index(drop=True)
    
    pd = subcellular.MorphologyPathDistanceCalculator(morph.to_morphio())
    soma = pandas.DataFrame({
        "afferent_section_id": [0],
        "afferent_segment_id": [0],
        "afferent_segment_offset": [0]
    })
    pt_df["soma_distance"] = pd.path_distances(pt_df, soma)[:, 0]
    return pt_df


def attribute_mesh_area(xyz, vertices, faces):
    tree = KDTree(xyz)
    face_centers = vertices[faces].sum(axis=1) / 3
    mapping_d, nn_ids_face = tree.query(face_centers)

    v_ = vertices[faces] # faces X face_vertices X (x,y,z)
    o_ = v_[:, [1, 2], :] - v_[:, [0], :]
    a = (o_[:, 0, :] * o_[:, 0, :]).sum(axis=1) * (o_[:, 1, :] * o_[:, 1, :]).sum(axis=1)
    b = (o_[:, 0, :] * o_[:, 1, :]).sum(axis=1)
    b = b ** 2
    face_area = 0.5 * np.sqrt(a - b)

    face_df = pandas.DataFrame({
        "point_idx": nn_ids_face,
        "area": face_area,
        "distance": mapping_d[faces[:, 0]]
    })
    return face_df

def add_mesh_area(pt_df, mesh_vertices, mesh_faces, column_name="surface_area"):
    face_df_mesh = attribute_mesh_area(pt_df[["x", "y", "z"]].to_numpy(),
                                       mesh_vertices, mesh_faces)
    per_seg_total_area = face_df_mesh.groupby("point_idx")["area"].sum()
    pt_df[column_name] = per_seg_total_area.reindex(pt_df.index).fillna(0.0)


def spine_area_df(m):
    from tqdm import tqdm
    spine_areas = []
    for spine_id in tqdm(range(m.spines.spine_count)):
        try:
            a_neck = m.spines.centered_spine_mesh(spine_id, include_head=False).area
        except:
            a_neck = 0.0
        try:
            a_head = m.spines.centered_spine_mesh(spine_id, include_neck=False).area
        except:
            a_head = 0.0
        spine_areas.append((a_neck, a_head))
    spine_areas = pandas.DataFrame(spine_areas, columns=["neck_area", "head_area"])
    spine_df = pandas.concat([
        m.spines.spine_table[["afferent_section_id", "afferent_segment_id", "afferent_segment_offset"]],
        spine_areas], axis=1)
    return spine_df

def downline_of(sec):
    ret = [sec.id + 1]
    for child in sec.children:
        ret += downline_of(child)
    return ret

def smoothed(df, col_x, col_y, w=50.0):
    x_data = df[col_x].to_numpy().reshape((1, -1))
    y_data = df[col_y].to_numpy().reshape((1, -1))
    x_smpl = np.arange(0, df[col_x].max() + 2)
    w = norm(0.0, w).pdf(x_smpl.reshape((-1, 1)) - x_data) # smpl X data
    return x_smpl, ((y_data * w).sum(axis=1) / w.sum(axis=1)).flatten()


def interpolate_for_short_secs(morph, master_df, min_len=1.0, smooth_width=15.):
    master_cp = master_df.index[master_df["length"] > min_len]

    for nrt_id in range(len(morph.neurites)):
        nrt = downline_of(morph.neurites[nrt_id].root_node)
        nrt_ = master_cp.intersection(nrt)
        to_interp = np.setdiff1d(nrt, nrt_)
        if len(to_interp) > 0:
            for col in Y_OPTIONS:
                x, y = smoothed(master_df.loc[nrt_], "soma_distance", col, 15.)
                master_df.loc[to_interp, col] = y[master_df.loc[to_interp, "soma_distance"].astype(int)]

def pack_rgb(rgb_u8):
    """(N,3) uint8 → (N,) uint32  packed 0x00RRGGBB"""
    r = rgb_u8[:, 0].astype(np.uint32)
    g = rgb_u8[:, 1].astype(np.uint32)
    b = rgb_u8[:, 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b
