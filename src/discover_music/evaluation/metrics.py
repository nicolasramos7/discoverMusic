import math

def recall_at_k(ranked_ids, relevant_ids, k):
    top = ranked_ids[:k]    #get top k ranked ids
    hits = len(set(top) & set(relevant_ids))    #out of those k, how many are in the relevant ids (we convert set to easily do set computations)
    return hits / len(relevant_ids) #return as a % of total relevant ids

def hit_rate_at_k(ranked_ids, relevant_ids, k):
    return 1.0 if set(ranked_ids[:k]) & set(relevant_ids) else 0.0  #same as above but yes/no

def ndcg_at_k(ranked_ids, relevant_ids, k): #explanation of formula in .txt fil
    rel = set(relevant_ids) #first we create a set of the most relevant ids for faster searching 
    #discount cumulative gain of actual result
    dcg = sum(1 / math.log2(pos + 1)    #for all do this math (equation)
              for pos, tid in enumerate(ranked_ids[:k], start=1)    #tid is the song id in question, and pos is the counter (starting at 1)
              if tid in rel)    #if tid (curr song) is in rel, apply to formula
    ideal_hits = min(k, len(rel))   #get min between the amount of relevant songs or k for the ideal hits constant
    idcg = sum(1 / math.log2(pos + 1) for pos in range(1, ideal_hits + 1))  #calc the ideal one (based on actual ranking)
    return dcg / idcg if idcg > 0 else 0.0  #return the ratio