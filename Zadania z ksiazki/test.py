

def something():
    return [[j for _ in range(1,3)]for j in range(0,10)]

# print(something())

tablica = []

for i in range(0,10):
    inner_list = []

    for _ in range(1,3):
        inner_list.append(i)   
    tablica.append(inner_list)

# print(tablica)

matrix = [[x 
           for _ in range(4)]
             for x in range(3)]
# print(matrix)

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


# def number_of_friends(user):
#     user_id = user["id"]
#     friend_ids = friendships[user_id]
#     return len(friend_ids)

books = {
    "book 1": 
    {
        "title": "Harry Potter and the Philosopher's Stone",
        "author": "J.K. Rowling",
        "year": 1997,
        "pages": 223
    },

    "book 2":
    {
        "title": "The Hobbit",
        "author": "J.R.R. Tolkien",
        "year": 1937,
        "pages": 310
    }
}

    # #Dodawanie elementu
# new_book = [('book 3', {"title" :"Witcher","author": "ASSndrzej Sapkowski", "year": 1989,"pages": 356})]
# books.update(new_book)

    # #Usuwanie elementu z slownika w slowniku
# books["book 1"].pop("title")

    # #Wyswietlanie slownika w slowniku
# def print_book_details(book):
#     for key, value in book.items():
#         print(f"{key}: {value}")
#     print() 


# for book_key, book in books.items():
#     print(f"Details of {book_key}:")
#     print_book_details(book)


colors = ["red", "blue", "green", "red", "blue", "yellow", "red", "yellow"]

my_dict = {}

for i in colors:
    if i in my_dict:
        my_dict[i] += 1
    else:
        my_dict[i] = 1
        
        
    
# print(my_dict)




dupa = ["red", "blue", "green", "red", "blue", "yellow", "red", "red", "red", "blue", "yellow", "red", "blue", "yellow", "red", "blue", "yellow", "red", "blue", "yellow", "red", "blue", "yellow"]

new_dict = {}

for i in dupa:
    if i in new_dict:
        new_dict[i] += 1
    else:
        new_dict[i] = 1



# print(new_dict)


students = {
    "student1": {
        "first_name": "Anna",
        "last_name": "Nowak",
        "age": 20,
        "subjects": ["Mathematics", "Physics", "Siku"]
    },
    "student2": {
        "first_name": "Buch",
        "last_name": "Blantowski",
        "age": 12,
       "subjects": ["Mathematics", "Sranie", "Computer Science"],
    }
}

new_student =[("student3", {"first_name": "Joint",
                            "last_name": "Kowaleweesdada",
                            "age": 28,
                            "subjects": ["Kupa", "Siku", "Sranie"]})]
students.update(new_student)

# def list_of_stedents(student):
#     for i, j in student.items():
#         print(f"{i}: {j}")
#     print()


# for key , value in students.items():
#     students[key].pop("age")
#     print(f"Details of student: {key}")
#     list_of_stedents(value)

# subject_of_student = {}

# for key, value in students.items():
#      print()
#      x = value['subjects']
#      for subject in x:
#         if subject not in subject_of_student:
#             subject_of_student[subject] = [key]
#         else:
#             subject_of_student[subject].append(key)

# for subject, student_keys in subject_of_student.items():
#     student_name = [students[student_key]["first_name"] for student_key in student_keys]
#     print(f"student of {subject}: {student_name}")
