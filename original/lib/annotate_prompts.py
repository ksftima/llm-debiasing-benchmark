system_prompts = {
    "amazon": "You are a perfect sentiment classification system",
    "misinfo": "You are a perfect newspaper article classification system",
    "biobias": "You are a perfect biography classification system",
    "germeval": "You are a perfect German tweet classification system",
}

dataset_labels = {
    "amazon": ["NEGATIVE", "POSITIVE"],
    "misinfo": ["THESUN", "THEGUARDIAN"],
    "biobias": ["MALE", "FEMALE"],
    "germeval": ["OTHER", "OFFENSIVE"],
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

def amazon_prompt(text, examples):
    return f"""
Classify the following review as either:
- POSITIVE if the review indicates an overall positive sentiment
- NEGATIVE if the review indicates an overall negative sentiment

Give no other explanation for your classification, only output the label.

Here are two examples of the formatting I would like you use, where <
REVIEW_TEXT > is a stand-in for the article text:

< REVIEW_TEXT >

CLASSIFICATION: POSITIVE

< REVIEW_TEXT >

CLASSIFICATION: NEGATIVE

{make_examples(examples)}

Here's the review to classify:

{text}

CLASSIFICATION: 
"""

def misinfo_prompt(text, examples):
    return f"""
Classify the following article as either:
- THESUN if it is likely to have been published in the British tabloid newspaper
  The Sun
- THEGUARDIAN if it is likely to have been published in the British daily
  newspaper The Guardian

Give no other explanation for your classification, only output the label.

Here are two examples of the formatting I would like you use, where <
ARTICLE_TEXT > is a stand-in for the article text:

< ARTICLE_TEXT >

CLASSIFICATION: THESUN

< ARTICLE_TEXT >

CLASSIFICATION: THEGUARDIAN

{make_examples(examples)}

Here's the article I would like you to classify:

{text}

CLASSIFICATION: 
"""

def biobias_prompt(text, examples):
    return f"""
Classify the following textual biographies as either:
- MALE if the subject is likely to be male
- FEMALE if the subject is likely to be female

Give no other explanation for your classification, only output the label.

Here are two examples of the formatting I would like you use, where < BIOGRAPHY_TEXT >
is a stand-in for the textual biography:

< BIOGRAPHY_TEXT >

CLASSIFICATION: MALE

< BIOGRAPHY_TEXT >

CLASSIFICATION: FEMALE

{make_examples(examples)}

Here's the textual biography I would like you to classify:

{text}

CLASSIFICATION: 
"""

def germeval_prompt(text, examples):
    return f"""
Classify the following German tweets as either:
- OFFENSIVE if the tweet is likely to contain an offense or be offensive
- OTHER if the tweet is _not_ likely to contain an offense of be offensive

Give no other explanation for your classification, only output the label.

Here are two examples of the formatting I would like you use, where < TWEET_TEXT >
is a stand-in for the text of the tweet:

< TWEET_TEXT >

CLASSIFICATION: OFFENSIVE

< BIOGRAPHY_TEXT >

CLASSIFICATION: OTHER

{make_examples(examples)}

Here's the German tweet I would like you to classify:

{text}

CLASSIFICATION: 
"""

def make_user_prompt(dataset, text, examples):
    if dataset == "amazon":
        return amazon_prompt(text, examples)
    elif dataset == "misinfo":
        return misinfo_prompt(text, examples)
    elif dataset == "biobias":
        return biobias_prompt(text, examples)
    elif dataset == "germeval":
        return germeval_prompt(text, examples)
    else:
        raise ValueError(f"'{dataset}' is not one of the known datasets.")

if __name__ == "__main__":

    examples = [
        ("I love you", "POSITIVE"),
        ("I hate you", "NEGATIVE"),
    ]

    print(amazon_prompt("I kind of like you", examples))
    print(misinfo_prompt("I kind of like you", examples))
    print(biobias_prompt("I kind of like you", examples))
    print(germeval_prompt("I kind of like you", examples))
