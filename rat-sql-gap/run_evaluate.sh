# run stanford corenlp
cd third_party/stanford-corenlp-full-2018-10-05
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 8999 -timeout 15000 > server.log &
cd ../../
# run evaluate
python run_evaluate.py preprocess experiments/sparc-tcs-configs-submit/gap-run.jsonnet
python run_evaluate.py eval experiments/sparc-tcs-configs-submit/gap-run.jsonnet

#cl run :configs :experiments :preprocess :pretrained_checkpoint :seq2struct :static :third_party :crash_on_ipy.py :nltk_data :run_evaluate.py checkpoint:checkpoint1 :sparc :run_evaluate.sh "sh run_evaluate.sh" --request-network --request-docker-image juruomagicpoi/gap-env:test --request-memory 8g --request-gpus 1