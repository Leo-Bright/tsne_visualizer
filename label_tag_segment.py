import json


def label_embeddings(embeddings, output, seg2tag, labels):

    seg2emb = {}
    tag_conut = {}

    normal_count = 0
    unnormal_count = 0

    with open(embeddings) as f:
        for line in f:
            segid_vector = line.strip().split()
            segid, seg_vec = segid_vector[0], segid_vector[1:]
            if len(seg_vec) < 3:
                continue
            seg2emb[segid] = seg_vec

    with open(output, 'w+') as f:
        for key in seg2emb:
            emb = seg2emb[key]

            if key in seg2tag and seg2tag[key] in labels:
                tag = seg2tag[key]
                f.write(key + ' %s' % ' '.join(emb) + ' ' + tag + '\n')
                if tag not in tag_conut:
                    tag_conut[tag] = 0
                tag_conut[tag] += 1
                normal_count += 1
            elif unnormal_count < (normal_count + 1)/(len(labels)-1):
                f.write(key + ' %s' % ' '.join(emb) + ' unnormal\n')
                unnormal_count += 1

    for key in tag_conut:
        print(key + " count: ", str(tag_conut[key]))
    print('unnormal count:', unnormal_count)


def gen_seg_tag_dict(tag_file_path, label_set):

    seg2tag = {}

    with open(tag_file_path) as f:

        seg_to_all_tag = json.loads(f.readline())

        for key in seg_to_all_tag:
            tag = seg_to_all_tag[key]
            if tag in label_set:
                seg2tag[key] = tag

    return seg2tag


if __name__ == '__main__':

    labels = ('Ave', 'unnormal')
    src_emb_file_path = 'data/segment/sanfrancisco_raw_feature_segment.embeddings'
    labeled_emb_path = 'data/segment/sanfrancisco_raw_feature_segment_labeled.embeddings'

    seg_tag_dict = gen_seg_tag_dict('data/segment/sf_segments_tiger_nametype.json', labels)

    label_embeddings(src_emb_file_path, labeled_emb_path, seg_tag_dict, labels)
