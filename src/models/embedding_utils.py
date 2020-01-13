import pickle
import numpy as np

def get_glove():
        # load dict here 
        dict_path = '../res/filtered_dict.pkl'
        with open(dict_path, 'rb') as f:
            search_terms_dict = pickle.load(f)
        
        glove_embedding = list(search_terms_dict.values())

        glove_embedding = np.array(glove_embedding)
        print('glove shape: ', glove_embedding.shape)
        return glove_embedding

def get_intent():
    dict_path = '../intent.pick'
    word_path = '../current_terms.pick'

    with open(dict_path, 'rb') as f:
        dict_list  = pickle.load(f)

    with open(word_path, 'rb') as f:
        word_list  = pickle.load(f)
    
    vecs = np.array([d['vector'] for d in dict_list])
    words = np.array([d['query'] for d in dict_list])
    
    embed_dict = dict(zip(words, vecs))
    
    ordered_embed = []
    for word in word_list:
        ordered_embed.append(embed_dict[word])

    ordered_embed = np.array(ordered_embed)
    print(ordered_embed.shape)
    return ordered_embed


def get_glove_and_intent():
    dict_path = '../res/filtered_dict.pkl'
    with open(dict_path, 'rb') as f:
        search_terms_dict = pickle.load(f)

    dict_path = '../intent.pick'
    with open(dict_path, 'rb') as f:
        dict_list = pickle.load(f)

    vecs = np.array([d['vector'] for d in dict_list])
    words = np.array([d['query'] for d in dict_list])
    
    embed_dict = dict(zip(words, vecs))

    word_path = '../current_terms.pick'
    with open(word_path, 'rb') as f:
        word_list = pickle.load(f)
    # Ensure that word order in glove and intend embedding matches that of search interest data.
    ordered_embed = []
    for word in word_list:
        if word in search_terms_dict:
            ordered_embed.append(np.concatenate([search_terms_dict[word], embed_dict[word]], axis=0))
    ordered_embed = np.array(ordered_embed)
    ordered_embed = (ordered_embed - np.mean(ordered_embed, axis=0)) / np.std(ordered_embed, axis=0)
    print(ordered_embed.shape)
    return ordered_embed

