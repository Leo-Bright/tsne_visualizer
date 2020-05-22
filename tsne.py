import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def save_embeddings(embeddings, output_file_path):
    with open(output_file_path, 'w+') as f:
        for id_emb in embeddings:
            f.write(id_emb[0])
            f.write(' '.join(map(str, id_emb[1])))
            f.write('\n')


if __name__ == '__main__':

    emb_file_path = 'data/sanfrancisco_raw_feature_crossing.embeddings'
    tsne_emb_path = 'data/sanfrancisco_raw_feature_crossing_tsne.embeddings'
    # df = pd.read_csv(emb_file_path)
    df = pd.read_csv(emb_file_path, header=None, sep=' ', index_col=0)
    X = df

    X_embedded = TSNE(n_components=2).fit_transform(X)
    print(X_embedded.shape)
    print(X_embedded)
    save_embeddings(X_embedded, tsne_emb_path)