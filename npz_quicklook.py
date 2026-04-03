import numpy as np




def query_entropRecord():
    data = np.load("data/entropy_records_gpt2-small_base_vs_contrast_n50.npz",
                    allow_pickle=True)
    print(data.files)

#    arr = data["norm_keys"]
#    print(arr)

#    arr = data["surfaces"]
#    print(arr[500,4,:])

    print(f"-------------------------------")
    print("unique norm_keys:", np.unique(data["norm_keys"]))
    print("unique hook_types:", np.unique(data["hook_types"]))
    print("unique alphas:", np.unique(data["alphas"]))
    print("unique roles:", np.unique(data["roles"]))

    # inspect first record of each key
    for k in data.files:
        arr = data[k]
        print(f"{k}: dtype={arr.dtype}, shape={arr.shape}, sample={arr.flat[0]}")

    for key in data.files:
        arr = data[key]
        print(f"  {key}, ", arr.shape)


def query_ablationRecord():
    data = np.load("data/ablation_records_gpt2-small_base_vs_contrast_n50.npz",
               allow_pickle=True)
    print("keys:", data.files)

    # inspect first record of each key
    for k in data.files:
        arr = data[k]
        print(f"{k}: dtype={arr.dtype}, shape={arr.shape}, sample={arr.flat[0]}")
    
def query_wuRecord():
    data = np.load("data/wu_subspace_records_gpt2-small_base_vs_contrast_n50.npz",
                   allow_pickle=True)
    print("keys:", data.files)
    for k in data.files:
        arr = data[k]
        print(f"{k}: dtype={arr.dtype}, shape={arr.shape}, sample={arr.flat[0]}")        

    print("unique norm_keys:", np.unique(data["norm_keys"]))
    print("unique alphas:",    np.unique(data["alphas"]))
    print("unique hook_types:", np.unique(data["hook_types"]))

    # what does the third dimension of surfaces represent?
    # check one record's surface shape vs its seq_len
    i = 0
    print(f"\nrecord 0: seq_len={data['seq_lens'][i]}, "
          f"n_layers={data['n_layers'][i]}, "
          f"surface shape={data['surfaces'][i].shape}")
    print(f"surface[0] (layer 0):", data['surfaces'][i, 0, :])


def main():
    
    print(f"\nQuery Entropy Record")
    query_entropRecord()

    print(f"\nQuery Ablation Record")
    query_ablationRecord()

    print(f"\nQuery WU Subspace Record")
    query_wuRecord()

   

if __name__ == "__main__":
    main()
