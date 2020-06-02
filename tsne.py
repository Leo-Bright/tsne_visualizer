import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def save_embeddings(embeddings, label, output_file_path):

    with open(output_file_path, 'w+') as f:
        idx = 0
        for emb in embeddings:
            f.write(' '.join(map(str, emb)))
            f.write(' ' + str(label[idx]))
            f.write('\n')
            idx += 1


if __name__ == '__main__':

    emb_file_path = 'data/segment/sanfrancisco_raw_feature_segment_labeled.embeddings'
    tsne_emb_path = 'data/segment/sanfrancisco_raw_feature_segment_labeled_tsne.embeddings'
    # df = pd.read_csv(emb_file_path)
    df = pd.read_csv(emb_file_path, header=None, sep=' ', index_col=0)

    rows, cols = df.shape
    label = df[cols]
    del df[cols]

    X = df

    X_embedded = TSNE(n_components=2).fit_transform(X)

    save_embeddings(X_embedded, label.values, tsne_emb_path)