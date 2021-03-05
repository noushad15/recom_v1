import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request, json
import requests
import math
import time
import sys

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
############################################ priorities #####################################
content_priority = 0.3
cat_priority = 0.4
module_priority = 0.2
weight_pririty = 0.2

##############################################################################
hdr = {'User-Agent':'Mozilla/5.0'}
tx  = "trainingexpress.org.uk"

################### urls #########################################
tx_bought_url = "https://trainingexpress.org.uk/wp-json/staffasia/v1/bought-together/"
tx_all_course_url = "https://trainingexpress.org.uk/wp-json/staffasia/v1/courses"

################################################################
def bought_together_list(courseId,n, provider):
    if provider == tx:
        urlB = tx_bought_url+str(courseId)
    else:
        return None
    try:
        data = requests.get(urlB, headers=hdr)
        data = json.loads(data.content)
        bought = data["bought_together"]
        df = pd.DataFrame.from_dict(bought)
        if n == 0:
            return df.T
        else:
            return df.T.head(n)
    except:
        return None

def get_all_courses(provider):
    if provider == tx:
        urlC = tx_all_course_url

    else:
        return None
    try:
        data = requests.get(urlC, headers=hdr)

        data = json.loads(data.content)

        bought = data["data"]

        df = pd.DataFrame.from_dict(bought)
        return df
    except:
        return None


def tfidf_cosine(courseId, provider):
    global all_course
    # print(all_course)
    # all_course.to_csv("tx_allcourse.csv",index=False)
    # for i in all_course.tags[0]:
    #     print(i)
    all_course.tags = all_course.tags.map(lambda x: ",".join(x).replace(" ","").replace(","," ").strip().lower())
    # print(all_course.tags)
    # all_course.tags = all_course.tags.map(lambda x: x.replace("[", "").replace("]", "").replace("'", "").lower())
    #
    # Initialize an instance of tf-idf Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Generate the tf-idf vectors for the corpus
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_course.tags)
    # cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    idx = all_course.loc[all_course['id'] == courseId].index
    # Get the pairwsie similarity scores
    sim_scores = list(enumerate(cosine_sim[idx[0]]))
    # Sort the course based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim = []
    for i in sim_scores:
        if all_course.loc[i[0]].id != courseId:
            sim.append(all_course.loc[i[0]].id)

    return pd.DataFrame(sim).rename(columns={0:"id"})

def sim_cat_courseId(courseId):
    global all_course
    # print(all_course)
    # all_course.to_csv("tx_allcourse.csv",index=False)
    # for i in all_course.tags[0]:
    #     print(i)
    all_course.categories = all_course.categories.map(lambda x: ",".join(x).replace(" ", "").replace(",", " ").strip().lower())
    # print(all_course.tags)
    # all_course.tags = all_course.tags.map(lambda x: x.replace("[", "").replace("]", "").replace("'", "").lower())
    #
    # Initialize an instance of tf-idf Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Generate the tf-idf vectors for the corpus
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_course.categories)
    # cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    idx = all_course.loc[all_course['id'] == courseId].index
    # Get the pairwsie similarity scores
    sim_scores = list(enumerate(cosine_sim[idx[0]]))
    # Sort the course based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim = []
    for i in sim_scores:
        if all_course.loc[i[0]].id != courseId:
            sim.append(all_course.loc[i[0]].id)

    return pd.DataFrame(sim).rename(columns={0:"id"})
def weight_price_Sold(courseId):
    global all_course
    all_course.sold_times = all_course.sold_times.map(lambda x: int(x))
    all_course.sale_price = all_course.sale_price.map(lambda x: float(x.replace("","0")))

    # print(type(all_course['sold_times'][0]))
    v = all_course['sold_times']
    R = all_course['sale_price']
    C = all_course['sale_price'].mean()
    m = all_course['sold_times'].quantile(0.5)
    all_course['weighted_average'] = ((R * v) + (C * m)) / (v + m)
    # print(C)
    # print(all_course.sort_values('weighted_average', ascending=False)["id"].reset_index(drop=True))

def nModule_duration_dist(courseId):
    global all_course
    x1 = all_course.loc[all_course.id == courseId, 'number_of_modules'].values[0]
    y1 = all_course.loc[all_course.id == courseId, 'course_duration'].values[0]
    all_course.number_of_modules = all_course.number_of_modules.map(lambda x: int(x))
    all_course.course_duration = all_course.course_duration.map(lambda x: int(x))
    # x = all_course['number_of_modules']
    # y = all_course['course_duration']

    # for i in all_course:
    #     i["dist"] = math.sqrt((i.number_of_modules -x1)*(i.number_of_modules-x1) + (i.course_duration-y1)*(i.course_duration-y1))

    # all_course = math.sqrt((x -x1)*(x-x1) + (y-y1)*(y-y1))
    # print(all_course)
    dict = []
    for i in range(len(all_course.number_of_modules)):
        dict.append(math.sqrt((all_course.loc[i].number_of_modules -x1)*(all_course.loc[i].number_of_modules-x1) + (all_course.loc[i].course_duration-y1)*(all_course.loc[i].course_duration-y1)))
    all_course["dict"] = dict
    a = all_course.sort_values(by='dict', ascending=True, ignore_index=True)#.reset_index(drop=True)
    # print(a)
    return a[a.id != courseId]
    # return a
    # print(all_course.dict, all_course.id)

def recommend(courseId, provider):
    global all_course
    s1 = time.time()
    # bought_together = bought_together_list(courseId,5,provider)
    all_course = get_all_courses(provider)
    s1e = time.time()
    similar_course_id = tfidf_cosine(courseId, provider)
    similar_cat_id = sim_cat_courseId(courseId)
    weight_price_Sold(courseId)
    nmodule = nModule_duration_dist(courseId)
    # nModule_duration_dist(courseId)
    # print(all_course)

    # print(similar_course_id.head(10))
    # print(similar_cat_id.head(10))
    # print(nModule_duration_dist(courseId).id.head(10))
    # print(all_course)

    # print(bought_together)
    # print(all_course.head(5))
    # all_course = all_course.sort_values('weighted_average', ascending=False)
    # scalling = MinMaxScaler()

    all_course["weight_scaled"] = all_course["weighted_average"] /all_course["weighted_average"].abs().max()
    all_course = all_course[all_course.id != courseId]
    # x = pd.DataFrame(x)

    ######################################################################################################################
    weight = np.linspace(1, 0, all_course.shape[0])

    nmodule["module"] = weight
    similar_cat_id["sim_cat"] = weight
    similar_course_id["similar"] = weight

    d1d2 = pd.merge(nmodule[["id", "module"]], similar_cat_id, on="id")
    d1d2d3 = pd.merge(d1d2, similar_course_id, on="id")
    marge_all = pd.merge(d1d2d3, all_course[["id", "weight_scaled"]], on="id")
    marge_all["final_score"] = marge_all.module*module_priority + marge_all.sim_cat*content_priority + marge_all.sim_cat*cat_priority + marge_all.weight_scaled*weight_pririty
    marge_all = marge_all.sort_values("final_score",ascending=False,ignore_index=True)

    ######################################################################################################################
    s2e = time.time()
    # print(all_course["weight_scaled"])
    # print(similar_course_id.head(5))
    # # print(nmodule.head(5))
    # print(similar_cat_id.head(5))
    # print(bought_together)
    print(marge_all.id.to_json())
    # print(marge_all.shape)


    # print("Data Response time: " + str(s1e - s1))
    # print("Process time: " + str(s2e - s1e))
    #


# courseId = int(input())
# provider = input()
cp = sys.argv[1]
courseId,provider = cp.split("_")

print("cp")

# courseId = int(courseId)
# recommend(courseId, provider)


# df = get_all_courses("tx")
# df.to_csv("tx_all_course.csv",index=False)
# print(df)