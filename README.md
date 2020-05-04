# Page_Rank
A Python Code to Pre-process -> Generate Graph -> Calculate Page Rank -> Calculate MRR for abstracts from WWW data
Note: !-- Do not extract the zip file --! !-- The complete program takes around 1-2 mins to run, please wait until the results show up on the command line(the same can be found below) --!

#Description:-
A Python Code to Pre-process -> Generate Graph -> Calculate Page Rank -> Calculate MRR for abstracts from WWW data

#Packages Used:-
1. numpy 
2. collections
3. re
4. zipfile
5. nltk (pip install --user -U nltk ::: 'https://www.nltk.org/install.html')
6. sys
7. pandas
8. Spacy (pip install -U spacy)
Also please make sure the language model is download for spacy using the following command: (python -m spacy download en_core_web_sm)
9. Networkx (pip install networkx)

#Instructions to Run the Code:-

Command:    python3 <script_name.py> <corups.zip> <window_size>
            
            python3 abalas26_hw4.py www.zip 6

            argv[0] : Name of the script file/Code
            argv[1] : The zip file of the documents to be processed [Note: Do not extract the file, please use the zip file as it reads the file using script]
            argv[2] : Window Size

1. Make sure that the code, the zip file of the documents are stored in the same directory or use absolute paths
2. View the answers for the questions of the assignment printed in the command line and also find the out at the output section of this file
3. Wait for the program to complete all stages and the program terminates automatically after the program is executed, no need to explicitly close connections
4. If testing with different data, please make sure to use the same folder structure, a zip file of abstracts and golds.

#Notes:-
1. The documents are retrieved from the zip file directory and the contents are stored in a dataframe for further processing
2. Tokenizing is done using Spacy library provided word_tokenizer and Spacy inbuilt punctuations are used to remove punctuations if any
3. The stop-words from  the Spacy library are used to remove the stop words and the tokens are stemmed using the NLTK provided Porter Stemmer library
4. The tokens with length less than 2 characters are  removed manually
5. All pre-processing are performed by a single function
6. Steps [1-5] are performed for both abstracts and golds
7. For each abstract document, a graph is generated and with a user provided window size, edges are added between the words in the window. 
   A final graph is generated. If the phrase is repeated multiple times, the edge weights are incremented by 1 (initial edge weight = 0)
8. From the graph generated, a matrix is obtained is reference and stored in the dataframe
9. From the Graph obtained, the page rank algorithm is run for each document and the page rank corresponding each node i.e word is calculated 
   and a dictionary corresponding to these are returned and stored in the dataframe
10. A list of n_grams are generated using the NLTK provided library, Also make note that n can be defined manually, if needed to experiment with
    different values of n like 4_gram or 6_gram, please check the code in the initialization section and proceed to change it.
11. The generated n_grams are ranked based on the page ranks obtained in step  9 and the phrases scored for all n_grams. If any word in the phrase
    doesn't occur in the page-rank, a score of 0 is added, this is similar to assigning page rank to adjacent words.
12. The MRR values for k in range of 1 to 10 is computed, also note that k value is user_defined, in the initialization section has the option to choose
    the value of k. 
13. The MRR's for k = 1 to 10 obtained in Step [12] for each files are taken and a mean for each document is taken and provided as the final output.
    The results for which are below. The results correspond to a window size of 6
14. If testing with other data, please make sure to follow the folder structure as this code reads zip file with the folder names abstract and gold. Please use same names and zip the files

#Output:-

MRR for top 1 n_grams with window size 6 : 0.045112781954887216
MRR for top 2 n_grams with window size 6 : 0.06390977443609022
MRR for top 3 n_grams with window size 6 : 0.07944862155388464
MRR for top 4 n_grams with window size 6 : 0.09411027568922296
MRR for top 5 n_grams with window size 6 : 0.10779448621553876
MRR for top 6 n_grams with window size 6 : 0.11781954887218031
MRR for top 7 n_grams with window size 6 : 0.12490870032223388
MRR for top 8 n_grams with window size 6 : 0.12923200859291054
MRR for top 9 n_grams with window size 6 : 0.13407745554362066
MRR for top 10 n_grams with window size 6 : 0.1376112901300867

Note:
Please not that using different stop words and tokenization method may lead to slight changes in the values, i.e in the order-of 0.0x. 
The Spacy provided stop words and tokenizer are considered to be highly optimized.
