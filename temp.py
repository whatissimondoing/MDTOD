import copy
import os
from tqdm import tqdm
from utils.io_utils import load_json, save_json

import torch
from transformers import PhrasalConstraint, AutoTokenizer, AutoModelForSeq2SeqLM


def paraphrase():
    def get_response(input_text, num_return_sequences, num_beams, constraints=None):
        batch = tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(torch_device)
        translated = model.generate(**batch,
                                    max_length=40,
                                    num_beams=num_beams,
                                    # num_beam_groups=num_return_sequences,  # False for constraints
                                    num_return_sequences=num_return_sequences,
                                    early_stopping=True,
                                    do_sample=True,  # False for constraints generation
                                    top_p=0.9,
                                    top_k=50,
                                    repetition_penalty=1.2,
                                    no_repeat_ngram_size=3,
                                    # constraints=constraints,
                                    temperature=0.9)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    model_name = 'stanford-oval/paraphraser-bart-large'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(torch_device)

    num_beams = 10
    num_return_sequences = 3
    context = "where is the nearest gas station ?"
    constraints = None
    # constraints = [
    #     PhrasalConstraint(
    #         tokenizer("city centre north b and b", add_special_tokens=False).input_ids
    #     ),
    #     PhrasalConstraint(
    #         tokenizer("north", add_special_tokens=False).input_ids
    #     ),
    #     PhrasalConstraint(
    #         tokenizer("328a histon road", add_special_tokens=False).input_ids
    #     ),
    #     PhrasalConstraint(
    #         tokenizer("guesthouse", add_special_tokens=False).input_ids
    #     ),
    # ]

    paraphrased_texts = get_response(context, num_return_sequences, num_beams, constraints=constraints)

    for p_text in paraphrased_texts:
        print(p_text)


def get_response(model, tokenizer, input_text, num_return_sequences, num_beams, constraints=None):
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch = tokenizer(input_text, truncation=True, padding='longest', max_length=60, return_tensors="pt").to(torch_device)
    with torch.no_grad():
        translated = model.generate(**batch,
                                    max_length=40,
                                    num_beams=num_beams,
                                    # num_beam_groups=num_return_sequences,  # False for constraints
                                    num_return_sequences=num_return_sequences,
                                    early_stopping=True,
                                    do_sample=True,  # False for constraints generation
                                    top_p=0.9,
                                    top_k=50,
                                    repetition_penalty=1.2,
                                    no_repeat_ngram_size=3,
                                    constraints=constraints,
                                    temperature=1.5
                                    )
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


def paraphrase_data(version, num_beams, num_return_sequences):
    model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(torch_device)

    model.eval()

    data_dir = os.path.join("data/MultiWOZ_{}".format(version))
    if version == "2.0":
        file_name = "annotated_user_da_with_span_full.json"
    else:
        file_name = "data.json"
    data = load_json(os.path.join(data_dir, file_name))

    for fn, raw_dial in tqdm(list(data.items())):
        log_ls = [[], [], [], []]
        for log_idx, log in enumerate(raw_dial["log"]):
            if log_idx < 12:
                log_ls[0].append(log['text'])
            elif 12 <= log_idx < 24:
                log_ls[1].append(log['text'])
            elif 24 <= log_idx < 36:
                log_ls[2].append(log['text'])
            elif 36 <= log_idx < 48:
                log_ls[3].append(log['text'])

            # constraints = []
            # constraint_list = log["dialog_act"].values()
            # for multi_cons in constraint_list:
            #     for cons in multi_cons:
            #         if cons[0] == "none" and cons[1] == "none":
            #             continue
            #         elif cons[-1] == "do nt care":
            #             continue
            #         elif cons[-1] in ["?", "yes", "none"]:
            #             constraints.append(
            #                 PhrasalConstraint(
            #                     tokenizer(cons[0], add_special_tokens=False).input_ids
            #                 )
            #             )
            #         else:
            #             constraints.append(
            #                 PhrasalConstraint(
            #                     tokenizer(cons[-1], add_special_tokens=False).input_ids
            #                 )
            #             )
            #
            # if len(constraints) == 0:
            #     constraints = None

        constraints = None
        paraphrased_texts = []

        for s_log_ls in log_ls:
            if len(s_log_ls) > 0:
                paraphrased_texts += get_response(model, tokenizer, s_log_ls, num_return_sequences, num_beams, constraints=constraints)

        index_num = 0
        for log_idx, log in enumerate(raw_dial["log"]):
            for p_num in range(num_return_sequences):
                log["text_" + str(p_num)] = paraphrased_texts[index_num]
                index_num += 1

    save_json(data, os.path.join(data_dir, "paraphrased_data.json"))


def paraphrase_data_incar(num_beams, num_return_sequences):
    model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(torch_device)

    model.eval()

    data_dir = os.path.join("data/InCar")

    file_name = "train_data.json"
    data = load_json(os.path.join(data_dir, file_name))

    for fn, raw_dial in tqdm(list(data.items())[:10]):

        for log_idx, log in enumerate(raw_dial["log"]):
            user_input = log["user"]
            resp = log["resp"]

            paraphrased_texts = []
            paraphrased_texts1 = []

            constraints = None

            paraphrased_texts += get_response(model, tokenizer, user_input, num_return_sequences, num_beams, constraints=constraints)
            paraphrased_texts1 += get_response(model, tokenizer, resp, num_return_sequences, num_beams, constraints=constraints)

            index_num = 0
            for p_num in range(num_return_sequences):
                log["user_" + str(p_num)] = paraphrased_texts[index_num]
                log["resp_" + str(p_num)] = paraphrased_texts1[index_num]
                index_num += 1

    save_json(data, os.path.join(data_dir, "para_train_data.json"))


import random
from nltk.corpus import wordnet
from nltk.corpus import stopwords


# nltk.download('stopwords') # run it for the first time
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def eda_data(version, aug_num):
    def eda_RI(originalSentence, n):
        """
        Paper Methodology -> Find a random synonym of a random word in the sentence that is not a stop word.
                             Insert that synonym into a random position in the sentence. Do this n times
        originalSentence -> The sentence on which EDA is to be applied
        n -> The number of times the process has to be repeated
        """
        stops = set(stopwords.words('english'))
        splitSentence = list(originalSentence.split(" "))
        splitSentenceCopy = splitSentence.copy()
        # Since We Make Changes to The Original Sentence List The Indexes Change and Hence an initial copy proves useful to get values
        ls_nonStopWordIndexes = []
        for i in range(len(splitSentence)):
            if splitSentence[i].lower() not in stops:
                ls_nonStopWordIndexes.append(i)
        if (n > len(ls_nonStopWordIndexes)):
            raise Exception("The number of replacements exceeds the number of non stop word words")
        WordCount = len(splitSentence)
        for i in range(n):
            indexChosen = random.choice(ls_nonStopWordIndexes)
            ls_nonStopWordIndexes.remove(indexChosen)
            synonyms = []
            originalWord = splitSentenceCopy[indexChosen]
            for synset in wordnet.synsets(originalWord):
                for lemma in synset.lemmas():
                    if lemma.name() != originalWord:
                        synonyms.append(lemma.name())
            if (synonyms == []):
                continue
            splitSentence.insert(random.randint(0, WordCount - 1), random.choice(synonyms).replace('_', ' '))
        return " ".join(splitSentence)

    def eda_RS(originalSentence, n):
        """
        Paper Methodology -> Find a random synonym of a random word in the sentence that is not a stop word.
                             Insert that synonym into a random position in the sentence. Do this n times
        originalSentence -> The sentence on which EDA is to be applied
        n -> The number of times the process has to be repeated
        """
        splitSentence = list(originalSentence.split(" "))
        WordCount = len(splitSentence)
        for i in range(n):
            firstIndex = random.randint(0, WordCount - 1)
            secondIndex = random.randint(0, WordCount - 1)
            while (secondIndex == firstIndex and WordCount != 1):
                secondIndex = random.randint(0, WordCount - 1)
            splitSentence[firstIndex], splitSentence[secondIndex] = splitSentence[secondIndex], splitSentence[firstIndex]
        return " ".join(splitSentence)

    def eda_RD(originalSentence, p=0.05, constraints=None):
        """
        Paper Methodology -> Randomly remove each word in the sentence with probability p.
        originalSentence -> The sentence on which EDA is to be applied
        p -> Probability of a Word Being Removed
        """
        og = originalSentence
        if (p == 1):
            raise Exception("Always an Empty String Will Be Returned")
        if (p > 1 or p < 0):
            raise Exception("Improper Probability Value")
        splitSentence = list(originalSentence.split(" "))
        lsIndexesRemoved = []
        WordCount = len(splitSentence)
        for i in range(WordCount):
            randomDraw = random.random()
            if randomDraw <= p and splitSentence[i] not in constraints:
                lsIndexesRemoved.append(i)
        lsRetainingWords = []
        for i in range(len(splitSentence)):
            if i not in lsIndexesRemoved:
                lsRetainingWords.append(splitSentence[i])
        if (lsRetainingWords == []):
            return og
        return " ".join(lsRetainingWords)

    data_dir = os.path.join("data/MultiWOZ_{}".format(version))
    if version == "2.0":
        file_name = "annotated_user_da_with_span_full.json"
    else:
        file_name = "data.json"
    data = load_json(os.path.join(data_dir, file_name))

    alpha = 0.05
    for fn, raw_dial in tqdm(list(data.items())):

        for log_idx, log in enumerate(raw_dial["log"]):
            n = max(1, int(len(log["text"].split()) * alpha))

            constraints = []

            if "dialog_act" not in log:  # some dialogues don't hava dialogue action ,so we skip it.
                continue

            constraint_list = log["dialog_act"].values()
            for multi_cons in constraint_list:
                for cons in multi_cons:
                    if cons[0] == "none" and cons[1] == "none":
                        continue
                    elif cons[-1] == "do nt care":
                        continue
                    elif cons[-1] in ["?", "yes", "none"]:
                        constraints.append(cons[0])
                    else:
                        constraints.append(cons[-1])

            for an in range(aug_num):
                rnd_choice = random.randint(0, 2)
                if rnd_choice == 0:
                    try:
                        log["text_" + str(an)] = eda_RI(log["text"], n)
                    except Exception as e:
                        log["text_" + str(an)] = log["text"]
                        print(e, log["text"])
                elif rnd_choice == 1:
                    log["text_" + str(an)] = eda_RS(log["text"], n)
                else:
                    log["text_" + str(an)] = eda_RD(log["text"], alpha, constraints)

    save_json(data, os.path.join(data_dir, "eda_data.json"))


def syn_replace_data(version, aug_num):
    def eda_SR(originalSentence, n, constraints=None):
        """
        Paper Methodology -> Randomly choose n words from the sentence that are not stop words.
                             Replace each of these words with one of its synonyms chosen at random.
        originalSentence -> The sentence on which EDA is to be applied
        n -> The number of words to be chosen for random synonym replacement
        """
        stops = set(stopwords.words('english'))
        if constraints is not None:
            stops += constraints
        splitSentence = list(originalSentence.split(" "))
        splitSentenceCopy = splitSentence.copy()
        # Since We Make Changes to The Original Sentence List The Indexes Change and Hence an initial copy proves useful to get values
        ls_nonStopWordIndexes = []
        for i in range(len(splitSentence)):
            if splitSentence[i].lower() not in stops:
                ls_nonStopWordIndexes.append(i)
        if (n > len(ls_nonStopWordIndexes)):
            raise Exception("The number of replacements exceeds the number of non stop word words")
        for i in range(n):
            indexChosen = random.choice(ls_nonStopWordIndexes)
            ls_nonStopWordIndexes.remove(indexChosen)
            synonyms = []
            originalWord = splitSentenceCopy[indexChosen]
            for synset in wordnet.synsets(originalWord):
                for lemma in synset.lemmas():
                    if lemma.name() != originalWord:
                        synonyms.append(lemma.name())
            if (synonyms == []):
                continue
            splitSentence[indexChosen] = random.choice(synonyms).replace('_', ' ')
        return " ".join(splitSentence)

    data_dir = os.path.join("data/MultiWOZ_{}".format(version))
    if version == "2.0":
        file_name = "annotated_user_da_with_span_full.json"
    else:
        file_name = "data.json"
    data = load_json(os.path.join(data_dir, file_name))

    alpha = 0.05
    for fn, raw_dial in tqdm(list(data.items())):

        for log_idx, log in enumerate(raw_dial["log"]):
            n = max(1, int(len(log["text"].split()) * alpha))
            for an in range(aug_num):
                try:
                    log["text_" + str(an)] = eda_SR(log["text"], n)
                except Exception as e:
                    log["text_" + str(an)] = log["text"]
                    print(e, log["text"])

    save_json(data, os.path.join(data_dir, "syn_replace_data.json"))


def line_figure_of_dual():
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(6.8, 5.8))  # Create a figure containing two axes.
    ax[0].plot([1, 2, 3, 4, 5], [79.57, 80.24, 82.96, 85.60, 87.52],
               label='Para',
               marker="o",
               linewidth=2,
               )  # Plot some data on the axes.
    ax[0].plot([1, 2, 3, 4, 5], [80.29, 77.06, 81.79, 83.33, 85.23],
               label='EDA',
               marker="s",
               linewidth=2,
               )  # Plot some data on the axes.
    ax[0].plot([1, 2, 3, 4, 5], [80.96, 73.58, 78.13, 80.35, 71.70],
               label='SYN',
               marker="s",
               linewidth=2,
               )  # Plot some data on the axes.
    # ax[0].set_xlabel("Temper/Margin")
    ax[0].set_ylabel("Comb.", fontsize=15)
    ax[0].set_ylim(bottom=70, top=90)
    ax[0].legend(fontsize=15)
    ax[0].grid(True, linestyle='--', linewidth=1)

    # second figure
    ax[1].plot([0.1, 0.3, 0.5, 0.7], [87.68, 88.60, 88.35, 87.83],
               label='TripletMarginLoss',
               marker="o",
               linewidth=2,
               )  # Plot some data on the axes.
    ax[1].plot([0.1, 0.3, 0.5, 0.7], [87.50, 87.58, 87.88, 87.15],
               label='InfoNCE',
               marker="s",
               linewidth=2,
               )  # Plot some data on the axes.
    ax[1].set_xlabel("Temper/Margin", fontsize=15)
    ax[1].set_ylabel("Task Score", fontsize=15)
    ax[1].set_ylim(bottom=None, top=None)
    # ax[1].legend()
    ax[1].grid(True, linestyle='--', linewidth=1)

    fig.show()
    # fig.savefig("analysis_cl.pdf", format="pdf", bbox_inches="tight")


def paraphrase_incar(data_path, para_num):
    import re
    def parrot_paraphrase(input_sent, para_num):
        print("-" * 100)
        print("Input_phrase: ", input_sent)
        print("-" * 100)
        try:
            para_phrases = parrot.augment(input_phrase=input_sent,
                                          use_gpu=True,
                                          do_diverse=True,  # Enable this to get more diverse paraphrases
                                          max_return_phrases=5,
                                          adequacy_threshold=0.50,  # Lower this numbers if no paraphrases returned
                                          fluency_threshold=0.50)[:para_num]
            for para_phrase in para_phrases:
                print(para_phrase)
            return [para[0] for para in para_phrases]
        except:
            print("No paraphrases returned")
            return []

    def eda_SR(originalSentence, n, constraints=None):
        """
        Paper Methodology -> Randomly choose n words from the sentence that are not stop words.
                             Replace each of these words with one of its synonyms chosen at random.
        originalSentence -> The sentence on which EDA is to be applied
        n -> The number of words to be chosen for random synonym replacement
        """
        stops = set(stopwords.words('english'))
        if constraints is not None:
            stops.union(constraints)
        splitSentence = list(originalSentence.split(" "))
        splitSentenceCopy = splitSentence.copy()
        # Since We Make Changes to The Original Sentence List The Indexes Change and Hence an initial copy proves useful to get values
        ls_nonStopWordIndexes = []
        for i in range(len(splitSentence)):
            if splitSentence[i].lower() not in stops:
                ls_nonStopWordIndexes.append(i)
        if (n > len(ls_nonStopWordIndexes)):
            raise Exception("The number of replacements exceeds the number of non stop word words")
        for i in range(n):
            indexChosen = random.choice(ls_nonStopWordIndexes)
            ls_nonStopWordIndexes.remove(indexChosen)
            synonyms = []
            originalWord = splitSentenceCopy[indexChosen]
            for synset in wordnet.synsets(originalWord):
                for lemma in synset.lemmas():
                    if lemma.name() != originalWord:
                        synonyms.append(lemma.name())
            if (synonyms == []):
                continue
            splitSentence[indexChosen] = random.choice(synonyms).replace('_', ' ')
        return " ".join(splitSentence)

    def extract_items(string):
        # 匹配 `[*]` 的正则表达式
        pattern = r'\[(.*?)\]'
        # 使用 `findall` 方法提取所有匹配的内容
        items = re.findall(pattern, string)
        return items

    from parrot import Parrot
    parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

    data_name = "train_data.json"
    data = load_json(os.path.join(data_path, data_name))

    new_data = {}
    for fn, raw_dial in tqdm(list(data.items())):
        new_data[fn] = raw_dial
        paraphrased_list = {}
        for log_idx, log in enumerate(raw_dial["log"]):
            input_para_lists = parrot_paraphrase(log["user"], para_num)
            if len(input_para_lists) < para_num:
                print("生成数量不够para_num的数量")
                left_para_num = para_num - len(input_para_lists)
                input_para_lists += [log["user"]] * left_para_num
            n = max(1, int(len(log["resp"].split()) * 0.05))
            constraints = set(extract_items(log["resp"]))
            resp_para_lists = [eda_SR(log["resp"], n, constraints) for i in range(para_num)]
            print(resp_para_lists)
            paraphrased_list[log_idx] = (input_para_lists, resp_para_lists)

        for para in range(para_num):
            logs = []
            for log_idx, log in enumerate(raw_dial["log"]):
                new_log = copy.deepcopy(log)
                new_log["user"] = paraphrased_list[log_idx][0][para]
                new_log["resp"] = paraphrased_list[log_idx][1][para]
                logs.append(new_log)
            new_data[str(para) + "_" + fn] = {"goal": {}, "log": logs}

    save_json(new_data, os.path.join(data_path, "train_data.json"))


def temp():
    text = "how is the weather this week in inglewood ?"

    from parrot import Parrot
    parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

    def parrot_paraphrase(input_sent, para_num):
        print("-" * 100)
        print("Input_phrase: ", input_sent)
        print("-" * 100)

        try:
            para_phrases = parrot.augment(input_phrase=input_sent,
                                          use_gpu=True,
                                          do_diverse=True,  # Enable this to get more diverse paraphrases
                                          max_return_phrases=5,
                                          adequacy_threshold=0.50,  # Lower this numbers if no paraphrases returned
                                          fluency_threshold=0.50)[:para_num]
            for para_phrase in para_phrases:
                print(para_phrase)
            return [para[0] for para in para_phrases]
        except:
            print("No paraphrases returned")
            return []

    print(parrot_paraphrase(text, 5))


if __name__ == '__main__':
    # paraphrase()
    # paraphrase_data_incar(num_beams=10, num_return_sequences=5)

    # eda_data(version="2.1", aug_num=5)
    # syn_replace_data(version="2.1", aug_num=5)
    # line_figure_of_dual()
    # paraphrase_incar("data/Kvret/processed_1", para_num=3)
    temp()
