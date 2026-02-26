system_prompts = {
    "fomc": "You are a perfect monetary policy sentiment classification system",
    "pubmedqa": "You are a perfect biomedical question answering classification system",
    "cuad": "You are a perfect legal license grant clause classification system",
    "misogynistic": "You are a perfect misogyny detection classification system",
}

dataset_labels = {
    "fomc": ["0", "1", "2"],
    "pubmedqa": ["0", "1", "2"],
    "cuad": ["0", "1"],
    "misogynistic": ["0", "1"],
}

def make_examples(examples):
    if examples == None:
        return ""
    else:
        lines = [
            "---",
            "",
            "Here is a set of examples for the task:",
            "",
        ]
        for review, label in examples:
            lines.append(review)
            lines.append("")
            lines.append("CLASSIFICATION: " + label)
            lines.append("")
        lines.append("---")
        return "\n".join(lines)

def fomc_prompt(text, examples):
    return f"""
Evaluate the monetary policy stance expressed in the following statement from Federal 
Open Market Committee (FOMC) communications.

Classify the text as one of the following:

    - 0 if the statement is DOVISH (suggests or indicates future monetary policy easing, 
such as lower interest rates, concerns about slow growth, or desire to raise inflation)

    - 1 if the statement is HAWKISH (suggests or indicates future monetary policy tightening, 
such as higher interest rates, concerns about inflation, or strong economic growth)

    - 2 if the statement is NEUTRAL (contains neither stance, presents mixed tones, 
or is purely descriptive without clear policy implications)

Give no other explanation for your classification, only output the label.

Here's the text I would like you to classify:

{text}

CLASSIFICATION:
"""

def pubmedqa_prompt(text, examples):
    return f"""
Evaluate whether the following biomedical text adequately answers the question 
posed at its beginning.

Classify the text as one of the following:

    - 0 if the text does NOT provide a clear answer to the question
    - 1 if the text DOES provide a clear and direct answer to the question  
    - 2 if the text provides a partial, uncertain, or ambiguous answer to the question

Give no other explanation for your classification, only output the label.

Here's the text I would like you to classify:

{text}

CLASSIFICATION: 
"""

def cuad_prompt(text, examples):
    return f"""
Classify the following contract clauses as either:
    - 0 if the clause DOES NOT include a license grant
    - 1 if the clause DOES include a license grant

Give no other explanation for your classification, only output the label.

Here are two examples of the formatting I would like you use, where
< CLAUSE_TEXT > is a stand-in for the contract-clause text:

< CLAUSE_TEXT  >

CLASSIFICATION: 0

< CLAUSE_TEXT >

CLASSIFICATION: 1

Here's the clause I would like you to classify:

{text}

CLASSIFICATION:
 
"""

def misogynistic_prompt(text, examples):
    return f"""
Evaluate whether the following online post expresses misogynistic content.
Misogynistic content includes text that expresses hatred, hostility, prejudice, 
or discrimination against women as a group, or promotes harmful stereotypes and derogatory attitudes toward women.

Classify the text as one of the following:
    - 0 if the text is NOT misogynistic (neutral discussion, general content, 
    or respectful commentary about gender)
    - 1 if the text IS misogynistic (expresses hatred, prejudice, discrimination, 
    or derogatory attitudes toward women)

Give no other explanation for your classification, only output the label.

Here are two examples of the formatting I would like you to use, 
where <ONLINE_POST> is a stand-in for the Reddit post:

<ONLINE_POST>

CLASSIFICATION: 0

<ONLINE_POST>

CLASSIFICATION: 1

Here's the text I would like you to classify:

{text}

CLASSIFICATION:

"""

def make_user_prompt(dataset, text, examples):
    if dataset == "amazon":
        return fomc_prompt(text, examples)
    elif dataset == "misinfo":
        return pubmedqa_prompt(text, examples)
    elif dataset == "biobias":
        return cuad_prompt(text, examples)
    elif dataset == "germeval":
        return misogynistic_prompt(text, examples)
    else:
        raise ValueError(f"'{dataset}' is not one of the known datasets.")

if __name__ == "__main__":

    examples = [
        ("I love you", "POSITIVE"),
        ("I hate you", "NEGATIVE"),
    ]

    print(fomc_prompt("I kind of like you", examples))
    print(pubmedqa_prompt("I kind of like you", examples))
    print(cuad_prompt("I kind of like you", examples))
    print(misogynistic_prompt("I kind of like you", examples))