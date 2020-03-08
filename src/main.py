from src.ap_core import Ap
import numpy as np
from src.ap_data import load_edges, load_checkins
from tqdm import tqdm

ds_path = "../data/loc-gowalla_edges.txt"
chk_path = "../data/loc-gowalla_totalCheckins.txt"

csr_ds = load_edges(ds_path)
checkins_train, checkins_test = load_checkins(chk_path)
def get_intersect(gt, pred):
    gt = set(gt)
    pred = set(pred)
    return len(gt.intersection(pred))

ap = Ap()
if __name__ == "__main__":
    ap.fit(csr_ds, 3)
    topn_base = ap.predict(checkins_train)
    cluster_prec = 0
    base_prec = 0
    for user_id in tqdm(np.unique(checkins_test.user_id)):
        gt_locs = checkins_test.loc[checkins_test.user_id == user_id, "location_id"].values
        if user_id in checkins_train.user_id:
            train_locs = ap.predict(checkins_train, user_id)
            cluster_prec += get_intersect(gt_locs, train_locs)
            base_prec += get_intersect(gt_locs, topn_base)
            # test_locs = ap.predict(checkins_test, user_id)

    print(f"Base AP {base_prec}")
    print(f"Cluster AP {cluster_prec}")



