# Write a CLI for a simple program...
# Will be the Frontend of our program... 
import click
import tensorflow as tf
import numpy as np

@click.command()
@click.option('--e', prompt="entity", 
              prompt_required=True, 
              help="Use this flag to input your text-entity.")

@click.option('--t', 
              prompt="text", 
              prompt_required=True,
              help="Use this flag to input your text for sentiment prediction.")

def hello(e, t):
    """
    CLI-General Info\n
    -------------------------------------------------\n
    (1) This is an entity-text sentiment prediction CLI.\n
    (2) Parameters: --e: text-entity, --t: text. Both are string values.\n
    (3) Both parameters are needed for the model to perform its task.\n
    (4) Expected Input format: 'Amazon I had a bad customer experience.'\n
    (5) With the example, Amazon is the text-entity (--e).\n
    (6) I had a bad customer experience is the text (--t)."""
    
    # user-end model inference

    # load our model
    model = tf.keras.models.load_model("Sentiment-Model.tf")
    
    # get user input
    input = " ".join([e,t])

    # run model to execute
    sentiment = model.predict([input])

    # get the index with the best probability
    sentiment = np.argmax(sentiment)

    # output conditions
    if sentiment == 0:
        click.echo({
            "entity": e,
            "text": t,
            "sentiment": "Negative"
        })
    
    elif sentiment == 1:
        click.echo({
            "entity": e,
            "text": t,
            "sentiment": "Positive"
        })
    
    elif sentiment == 2:
        click.echo({
            "entity": e,
            "text": t,
            "sentiment": "Neutral"
        })
    
    else:
        click.echo({
            "entity": e,
            "text": t,
            "sentiment": "Irrelevant"
        })
    
    
if __name__ == '__main__':
    # click-cli function
    hello()