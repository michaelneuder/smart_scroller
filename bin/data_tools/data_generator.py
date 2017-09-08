#!/usr/bin/env python3
import random
from math import sqrt
import numpy as np

#---------------------------------------------------------------
#                     Data to be generated
# (1) number of phrases
# (2) number of total words
# (3) words per phrase (2/1)
# (4) number of non-function words
# (5) non-function words per phrase (4/1)
# (6) mean of total words
# (7) std. dev. of total words
# (8) mean of non-function words
# (9) std. dev. of non-function words
# (10) mean of ln(word frequency) (total words)
# (11) std. dev. of ln(word frequency) (total words)
# (12) mean of ln(word frequency) (non-function words)
# (13) std. dev. of ln(word frequency) (non-function words)
# constant term = 1
# scroll time for the current page based on a gamma dist.
#---------------------------------------------------------------

def generate_page_data():
    data = []

    for i in range(3):
        # generating (1) using range of 1-4 --- E(X) = 2.5
        num_sentences = random.randint(1,4)
        data.append(num_sentences)

        # generating (2) --- using range of 30-50 --- E(X) = 40
        num_words = random.randint(30,50)
        data.append(num_words)

        # generating (3) --- dividing (2)/(1) --- E(X) = 16
        data.append(round((num_words/num_sentences),4))

        # generating (4) --- using range of 15-25 --- E(X) = 20
        num_words_nf = random.randint(15,25)
        data.append(num_words_nf)

        # generating (5) --- dividing (4)/(1) --- E(X) = 8
        data.append(round((num_words_nf/num_sentences),4))

        # generating (6) --- using range of 3-8 --- E(X) = 5.5
        data.append(random.randint(3,8))

        # generating (7) --- using range of 1.5-2.5 --- E(X) = 2
        data.append(round(random.uniform(1.5,2.5),4))

        # generating (8) --- using range of 4-9 --- E(X) = 6.5
        data.append(random.randint(4,9))

        # generating (9) --- using range of 2-3 --- E(X) = 2.5
        data.append(round(random.uniform(2.0,3.0),4))

        # generating (10) --- using range of 11.2-16.6 --- E(X) = 13.9

        data.append(round(random.uniform(11.2,16.6),4))

        # generating (11) --- using range of 1.0-2.0 --- E(X) = 1.5
        data.append(round(random.uniform(1.0,2.0),4))

        # generating (12) --- using range of 11.2-15.0 --- E(X) = 13.1
        data.append(round(random.uniform(11.2,15.0),4))

        # generating (13) --- using range of 1.0-2.0 --- E(X) = 1.5
        data.append(round(random.uniform(1.0,2.0),4))

    # generating constant = 1
    data.append(1)

    # generating representation
    data.append(sum(data))

    return np.asarray(data)

def main():
    with open("synthetic_data.csv", mode='w') as WRITE_FILE:
        num_iterations = int(input("enter the number of iterations: "))
        for i in range(num_iterations):
            current_data = generate_page_data()
            current_data_str = ','.join(str(elements) for elements in current_data)
            print(current_data_str)
            WRITE_FILE.write(current_data_str + "\n")
    WRITE_FILE.close()

    print("---------------------------------------------------------")
    print("number of iterations producted", str(num_iterations))
    print("---------------------------------------------------------")

if __name__ == '__main__':
    main()
