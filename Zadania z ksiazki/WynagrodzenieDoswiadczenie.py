from collections import defaultdict
salaries_and_tenures = [(83000, 8.7), (88000, 8.1),
                        (48000, 7),(76000, 6),
                        (69000, 6.5),(76000, 7.5),
                        (60000, 2.5),(83000, 10),
                        (48000, 1.9),(63000, 4.2)]


salaries_by_tenures =defaultdict(list)

for salary, tenure in salaries_and_tenures:
    salaries_by_tenures[tenure].append(salary)

average_salary_by_tenure = {
    tenure: sum(salaries)/len(salaries)
    for tenure, salaries in salaries_by_tenures.items()
}

def tenure_bucket(tenure):
    if tenure <2 :
        return "Mniej niz 2 lata"
    elif tenure <5:
        return "pomiedzy 2 a 5 lat"
    else :
        return "powyzej 5 lat"
    
salaries_by_tenure_bucket = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salaries_by_tenure_bucket[bucket].append(salary)

average_salary_by_bucket = {
    tenure_bucket: round(sum(salaries)/len(salaries), 1)for tenure_bucket, salaries in salaries_by_tenure_bucket.items()
}

print(average_salary_by_bucket)
