import pickle
import numpy as np
from src.data.utils import read_query_from_file


def get_glove(filtered_dict_path):
        # load dict here
        with open(filtered_dict_path, 'rb') as f:
            search_terms_dict = pickle.load(f)
        
        glove_embedding = list(search_terms_dict.values())

        glove_embedding = np.array(glove_embedding)
        print('glove shape: ', glove_embedding.shape)
        return glove_embedding


def get_wv_dict(wv_dict_path, current_word_path):
    with open(wv_dict_path, 'rb') as fi:
        wv_dict = pickle.load(fi)

    word_list = read_query_from_file(current_word_path)

    ordered_embed = []
    for word in word_list:
        ordered_embed.append(wv_dict[word])

    ordered_embed = np.array(ordered_embed)
#     print(ordered_embed.shape)
    return ordered_embed


def get_glove_and_intent(filtered_dict_path, intent_dict_path, current_word_path):

    with open(filtered_dict_path, 'rb') as f:
        search_terms_dict = pickle.load(f)

    with open(intent_dict_path, 'rb') as f:
        dict_list = pickle.load(f)

    vecs = np.array([d['vector'] for d in dict_list])
    words = np.array([d['query'] for d in dict_list])
    
    embed_dict = dict(zip(words, vecs))

    word_list = read_query_from_file(current_word_path)

    # Ensure that word order in glove and intend embedding matches that of search interest data.
    ordered_embed = []
    for word in word_list:
        if word in search_terms_dict:
            ordered_embed.append(np.concatenate([search_terms_dict[word], embed_dict[word]], axis=0))
    ordered_embed = np.array(ordered_embed)
    ordered_embed = (ordered_embed - np.mean(ordered_embed, axis=0)) / np.std(ordered_embed, axis=0)
    # print(ordered_embed.shape)
    return ordered_embed

