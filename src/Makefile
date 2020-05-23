train:
	python main.py --train --model_name $(MODEL_NAME) --code_book_len $(M) --cluster_num $(K)

evaluate:
	python main.py --evaluate --model_name $(MODEL_NAME) 

run_classifier_glove:
	python run_classifier.py --evaluate --model_name $(MODEL_NAME) --glove_model_name $(GLOVE_MODEL_NAME)

run_classifier_glove_compressed:
	python run_classifier.py --evaluate --model_name $(MODEL_NAME) --glove_model_name $(GLOVE_MODEL_NAME) --use_word2code


