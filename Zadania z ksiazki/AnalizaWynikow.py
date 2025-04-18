students_scores = {
    "Anna": {
        "Mathematics": 4.5,
        "Physics": 5.0,
        "Chemistry": 4.0
    },
    "Buch": {
        "Mathematics": 3.5,
        "Physics": 4.0,
        "Chemistry": 2.5
    },
    "Joint": {
        "Mathematics": 5.0,
        "Physics": 4.5,
        "Chemistry": 4.5
    },
    "Katarzyna": {
        "Mathematics": 3.0,
        "Physics": 2.5,
        "Jaja": 4.5
    }
}


def calculate_average_scores(students_scores):
    calculate_average = {}
    for student, subjects in students_scores.items():
        total_score = 0
        count = 0
        for subjects, score in subjects.items():
            total_score += score
            count += 1
        
        if count > 0:
            average = total_score/count
            calculate_average[student] = round(average, 1)

    return calculate_average
            

print(calculate_average_scores(students_scores))

avreage_students_scores = calculate_average_scores(students_scores).items()



def get_students_above_average(avreage_students_scores, average):
    better_students = {}
    for stuednt , student_average in avreage_students_scores:
        if student_average > average:
            better_students[stuednt] = student_average
    
    return better_students

print(get_students_above_average(avreage_students_scores,4))



def highest_scores_per_subject(students_scores):
    highest_mark = {}
    for student, subjcets in students_scores.items():
        for subject, mark in subjcets.items():
            if subject not in highest_mark:
                highest_mark[subject] =  mark
            else:
                highest_mark[subject] = max(highest_mark[subject], mark)
        
    return highest_mark


print(highest_scores_per_subject(students_scores))

for i in [1,2,3,4,5]:
    print(i)