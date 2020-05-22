import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def save_embeddings(embeddings, label, output_file_path):

    with open(output_file_path, 'w+') as f:
        idx = 0
        for id_emb in embeddings:
            f.write(str(id_emb[0]))
            f.write(' '.join(map(str, id_emb[1])))
            f.write(' ' + label[idx])
            f.write('\n')
            idx += 1


if __name__ == '__main__':

    emb_file_path = 'data/sanfrancisco_raw_feature_crossing_labeled.embeddings'
    tsne_emb_path = 'data/sanfrancisco_raw_feature_crossing_tsne.embeddings'
    # df = pd.read_csv(emb_file_path)
    df = pd.read_csv(emb_file_path, header=None, sep=' ', index_col=0)
    X = df

    rows, cols = df.shape
    label = df[cols]
    del df[cols]

    X_embedded = TSNE(n_components=2).fit_transform(X)

    save_embeddings(X_embedded, label, tsne_emb_path)