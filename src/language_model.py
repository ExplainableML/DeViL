from transformers import AutoModelForCausalLM, AutoTokenizer


def get_language_model(nlp_model_name):
    # get tokenizer
    tokenizer = get_tokenizer(nlp_model_name)
    # get model
    language_model = AutoModelForCausalLM.from_pretrained(nlp_model_name)

    if "opt" in nlp_model_name:
        embedding_weights = language_model.model.decoder.embed_tokens
        embed_dim = language_model.config.word_embed_proj_dim
    elif "gpt2" in nlp_model_name:
        embedding_weights = language_model.transformer.wte
        embed_dim = language_model.config.n_embd
    elif "BERT" in nlp_model_name:
        language_model = AutoModelForCausalLM.from_pretrained(
            nlp_model_name, is_decoder=True
        )
        embedding_weights = language_model.bert.embeddings.word_embeddings
        embed_dim = language_model.config.hidden_size
    else:
        raise NotImplementedError

    return language_model, tokenizer, embedding_weights, embed_dim


def get_tokenizer(language_model_name):
    tokenizer = AutoTokenizer.from_pretrained(language_model_name, use_fast=False)
    if "BERT" not in language_model_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_bos_token = True
    return tokenizer
