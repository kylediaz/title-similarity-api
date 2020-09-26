import json

def public(request):
    """
    :param request:
    Contains JSON in this format:
    {
        threshold: [0, 1] threshold for articles to be "similar", optional
        articles: [
            {
                ID: arbitrary unique ID
                title: string
                body: string
            },
            ...
        ]
    }
    :return:
    Lists of articles that are similar.
    {
        error: boolean,
        error_message: string,
        like_articles: [
            [id1, id2, id3, ...],
            [id4, id5, id6, ...]
        ]
    }
    """
    request_json = request.get_json()
    if not request or not request_json or not request_json['articles']\
            or type(request_json['articles']) is not list\
            or (request_json['threshold'] and type(request_json['threshold']) is not float):
        return ':('
    articles = request_json['articles']
    threshold = request_json['threshold'] or .6

    like_articles = find_like_articles(articles, threshold)
    output = {
        'error': False,
        'error_message': '',
        'like_articles': like_articles
    }
    return json.dumps(output)


import itertools

def find_like_articles(articles: list, threshold: float) -> list:
    """
    :param articles: articles to compare
    :param threshold: [0, 1] like index where articles will be considered similar
    :return: lists of the IDs of articles that are similar
    """
    graph = dict()

    def add(addend, to) -> None:
        graph[to] = graph.get(to, [to])
        while not isinstance(graph[to], list):
            to = graph[to]
        graph[to].append(addend)
        graph[addend] = to

    def get(id) -> list:
        while not isinstance(graph[id], list):
            id = graph[id]
        return graph[id]

    for (A, B) in itertools.combinations(articles, r=2):
        print(graph)
        print(A, B)
        A_ID = A['ID']
        B_ID = B['ID']
        if A_ID in graph and B_ID in graph and graph[A_ID] == graph[B_ID]:
            continue
        sim = similarity(A['title'], B['title'])
        if sim >= threshold:
            if B_ID not in graph:
                add(B_ID, A_ID)
            elif A_ID not in graph and B_ID in graph:
                add(A_ID, B_ID)
            elif A_ID in graph and B_ID in graph and graph[A_ID] != graph[B_ID]:
                B_items = get(B_ID)
                for B_item in B_items:
                    add(B_item, A_ID)

    return get_connected_sub_graphs(graph)


def get_connected_sub_graphs(graph: dict) -> list:
    output = []
    for value in graph.values():
        if isinstance(value, list):
           output.append(value)
    return output


import os
import requests

dandelion_template = "https://api.dandelion.eu/datatxt/sim/v1/?text1=%s&text2=%s&token=" # + os.environ.get("dandelion_token")

def similarity(s1: str, s2: str) -> float:
    s1_embed_tensor = get_embed(s1)
    s2_embed_tensor = get_embed(s2)
    return cosine_similarity(s1_embed_tensor, s2_embed_tensor)
    # OLD
    request_string = dandelion_template % (s1, s2)
    r = requests.get(request_string)
    return r.json()['similarity']


from sentence_transformers import SentenceTransformer, util
import numpy as np

embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

def get_embed(s: str) -> float:
    return embedder.encode([s], convert_to_tensor=True)


def cosine_similarity(A, B) -> float:
    return util.pytorch_cos_sim(A, B)[0]