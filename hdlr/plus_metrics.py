import numpy as np

# def serendipity(y_true, y_pred, threshold):
#     """
#     Calcula a serendipidade de uma lista de recomendações previstas e uma lista de classificações reais.

#     Args:
#         y_true (array-like): Lista de classificações reais.
#         y_pred (array-like): Lista de recomendações previstas.
#         threshold (float): Limiar de relevância para considerar um item como relevante.

#     Returns:
#         A serendipidade calculada.
#     """
#     y_true = np.array(list(map(int, y_true)))
#     y_pred = np.array(list(map(int, y_pred)))
#     threshold = int(threshold)
    
#     # Filtra os itens relevantes na lista de classificações reais
#     relevant_items = np.where(y_true >= threshold)[0]

#     # Verifica se a lista de recomendações previstas contém algum item irrelevante
#     irrelevant_items = np.setdiff1d(np.arange(len(y_pred)), relevant_items)

#     # Calcula a serendipidade como a proporção de itens irrelevantes na lista de recomendações previstas
#     serendipity = len(irrelevant_items) / float(len(y_pred))

#     return serendipity

def serendipity(true_ratings, predicted_ratings, threshold):
    num_relevant_items = 0
    num_unexpected_items = 0
    
    for true_rating, predicted_rating in zip(true_ratings, predicted_ratings):
        if true_rating >= threshold:
            num_relevant_items += 1
            if predicted_rating < threshold:
                num_unexpected_items += 1
    
    serendipity = num_unexpected_items / num_relevant_items if num_relevant_items > 0 else 0
    return serendipity


from math import log

def diversity(y_true, y_pred):
    """
    Calcula a diversidade baseada no índice de Shannon.
    
    Args:
        y_true (list): Lista de valores verdadeiros.
        y_pred (list): Lista de valores preditos.
        
    Returns:
        float: Diversidade calculada.
    """
    # Concatena as listas y_true e y_pred para criar uma lista única
    data = y_true + y_pred
    
    # Conta a frequência de cada elemento na lista
    counts = {}
    for d in data:
        counts[d] = counts.get(d, 0) + 1
        
    # Calcula a diversidade
    n = len(data)
    diversity = 0.0
    for count in counts.values():
        p = count / n
        diversity -= p * log(p, 2)
        
    return diversity
