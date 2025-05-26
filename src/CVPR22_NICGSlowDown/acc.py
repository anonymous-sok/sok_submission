from utils import *
import torch
from train import corpus_bleu
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

BATCH = 20
max_length = 60
device = torch.device('cuda')


def get_ground_truth():
    results = []
    for i, data in enumerate(test_loader):
        (imgs, caption, caplen, all_captions) = data
        all_captions = [all_captions[i * CAP_PER_IMG].tolist() for i in range(BATCH)]

        for img_caps in all_captions:
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            results.append(img_captions)

        if i >= 10:
            break
    return results


rrrr = []
for model_file in MODEL_FILE_LIST:
    task_name = model_file.split('.')[0]
    encoder, decoder, test_loader, _, word_map = load_dataset_model(model_file, batch_size=BATCH * CAP_PER_IMG)

    ground_truth = get_ground_truth()
    print(task_name)

    for attack_name in ['L2', 'Linf']:
        print(attack_name)
        adv_file = 'adv/spatial/' + str(0) + '_' + attack_name + '_' + task_name + '.adv'
        res = torch.load(adv_file,weights_only=False)
        ori_seqs, adv_seqs = [], []
        for data in res:
            ori_img = data[0][0].to(device) 
            adv_img = data[1][0].to(device) 

            ori_seq, _ = prediction_batch(ori_img, encoder, decoder, word_map, max_length, device)
            for ori_s in ori_seq:
                ori_s = ori_s.tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        [ori_s]))  # remove <start> and pads
                ori_seqs.append(img_captions[0])

            adv_seq, _ = prediction_batch(adv_img, encoder, decoder, word_map, max_length, device)
            for adv_s in adv_seq:
                adv_s = adv_s.tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        [adv_s]))  # remove <start> and pads
                adv_seqs.append(img_captions[0])

        min_len = min(len(ground_truth), len(ori_seqs), len(adv_seqs))
        ground_truth = ground_truth[:min_len]
        ori_seqs = ori_seqs[:min_len]
        adv_seqs = adv_seqs[:min_len]

        #print("Ground truth count:", len(ground_truth))
        #print("Generated sequences:", len(ori_seqs))
        smooth = SmoothingFunction().method1  # method1 is a good default
        ori_bleu4 = corpus_bleu(ground_truth, ori_seqs, smoothing_function=smooth)
        adv_bleu4 = corpus_bleu(ground_truth, adv_seqs, smoothing_function=smooth)
        #ori_bleu4 = corpus_bleu(ground_truth, ori_seqs)
        #adv_bleu4 = corpus_bleu(ground_truth, adv_seqs)
        print(ori_bleu4, adv_bleu4)
        rrrr.append(np.array([ori_bleu4, adv_bleu4]).reshape([1, -1]))
rrrr = np.concatenate(rrrr, axis=0)
np.savetxt('acc.csv', rrrr, delimiter=',')




"""Copies bytes from a large (1GB) input 
      stream to an output stream.
      :param in_stream: input stream.
      :param out_stream: output stream."""
def copy_stream(in_stream, out_stream):
        while True:
            data = in_stream.read(1024)
            if not data:
                break
            out_stream.write(data)
