# Implementation of Naive Bayes Classifier #
## NOTE All codes are based on MachineLearnigAction ##

### This project consist of following functions ###

- Iporting a list of string from the exsiting documents
- Parse text into token vectors
- Train the bayes-classifier with these token vectors
- Use this classifier to determing if one e-mail is a spam e-mail

### Steps for classifying spam email with naive Bayes ###

> Example: using naïve Bayes to classify email
> 1. Collect: Text files provided.  
> 
> 2. Prepare: Parse text into token vectors.  
> 
> 3. Analyze: Inspect the tokens to make sure parsing was done correctly.  
> 
> 4. Train: Use trainNB0() that we created earlier.  
> 
> 5. Test: Use classifyNB() and create a new testing function to calculate the error
>    rate over a set of documents.  
>    
> 6. Use: Build a complete program that will classify a group of documents and print
>    misclassified documents to the screen.  
>    