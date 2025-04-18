users = [
    {"id": 0, "name": "Daniel"},
    {"id": 1, "name": "Anna"},
    {"id": 2, "name": "Jakub"},
    {"id": 3, "name": "Marta"},
    {"id": 4, "name": "Katarzyna"},
    {"id": 5, "name": "Piotr"},
    {"id": 6, "name": "Zuzanna"},
    {"id": 7, "name": "Tomasz"},
    {"id": 8, "name": "Agnieszka"},
    {"id": 9, "name": "Michał"}
]

friendship_pairs =[
    (0, 1), 
    (0, 2),
    (0, 4), 
    (1, 2), 
    (1, 3), 
    (2, 3), 
    (3, 4), 
    (4, 5), 
    (5, 6), 
    (5, 7), 
    (6, 8), 
    (7, 0), 
    (1, 9)
]

friendships = {user["id"]: [] for user in users}
# print(friendships)

for i,j in friendship_pairs:
    friendships[i].append(j)
    friendships[j].append(i)

# print(friendships)
# print()

def number_of_friends(user):
    user_id = user["id"]
    friend_ids = friendships[user_id]
    return len(friend_ids)

total_connetion = sum(number_of_friends(user) for user in users)

# print(total_connetion)


num_friends_by_id = [(user["id"], number_of_friends(user)) for user in users]
#print(num_friends_by_id)

num_friends_by_id.sort(
    key= lambda id_and_friends: id_and_friends[1],reverse=True)

# print(num_friends_by_id)

def foaf_ids_bad(user):
    return [foaf_id
             for friend_id in friendships[user["id"]]
                for foaf_id in friendships[friend_id]]

# print(foaf_ids_bad(users[0]))
# print(foaf_ids_bad(users[1]))
# print(foaf_ids_bad(users[2]))

from collections import Counter

def friend_of_friend_ids(user):
    user_id = user["id"]
    return Counter(
        users[foaf_id]["name"]
            for friend_id in friendships[user_id]
                for foaf_id in friendships[friend_id]
                    if foaf_id != user_id and foaf_id not in friendships[user_id]

    )

# print(friend_of_friend_ids(users[3]))

interests = [
    (0, "Big Data"),
    (0, "HBase"),
    (0, "Java"),
    (0, "Hadoop"),
    (0, "Spark"),
    (0, "Storm"),
    (0, "Cassandra"),
    (1, "NoSQL"),
    (1, "MongoDB"),
    (2, "Python"),
    (2, "scikit-learn"),
    (2, "SciPy"),
    (2, "NumPy"),
    (2, "statsmodels"),
    (2, "pandas"),
    (3, "R"),
    (3, "Python"),
    (4, "machine learning"),
    (4, "regression"),
    (5, "Java"),
    (5, "decision trees"),
    (5, "Haskell"),
    (5, "programming languages"),
    (6, "statistics"),
    (6, "probability"),
    (6, "mathematics"),
    (7, "machine learning"),
    (7, "scikit-learn"),
    (7, "Mahout"),
    (8, "neural networks"),
    (8, "deep learning"),
    (9, "Big Data"),
    (8, "artificial intelligence"),
    (9, "Hadoop"),
    (9, "Java"),
    (9, "MapReduce"),
    (9, "Big Data")
]

def data_scientists_who_like(target_intrest):
    return [user_id
                for user_id, user_intrest in interests
                if user_intrest == target_intrest
    ]

print(data_scientists_who_like("Java"))

from collections import defaultdict

user_ids_by_intrest =defaultdict(list)

for user_id, intrest in interests:
    user_ids_by_intrest[intrest].append(user_id)

# for intrest, user_ids in user_ids_by_intrest.items():
#      print(f"{intrest}: {user_ids}", end='\n')

interested_by_user_id = defaultdict(list)

for user_id, interest in interests:
    interested_by_user_id[interest].append(user_id)

# print(interested_by_user_id)

most_famous = []
for interest, user_ids in interested_by_user_id.items():
     most_famous.append((interest, len(user_ids)))

most_famous.sort(key=lambda x: x[1], reverse=True)

for interest, count in most_famous:
     print(f'{interest}: {count}')
    

# def most_common_interests_with(user):
#     user_interests = interested_by_user_id[user["id"]]
#     print(f"Interests for user {user['name']}: {user_interests}")

#     return Counter(
#         interested_user_id
#         for interest in interested_by_user_id[user["id"]]
#         for interested_user_id in user_ids_by_intrest[interest]
#         if interested_user_id != user["id"]
#     )

# result = most_common_interests_with(users[2])  
# print(result)






