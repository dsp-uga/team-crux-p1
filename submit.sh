python ./setup.py bdist_egg


gcloud dataproc jobs submit pyspark \
    --cluster team-crux-p1 \
    --py-files ./dist/team_crux_p1-1.0.0.dev0-py3.6.egg \
    ./main.py \
    -- \
    -v \
    --dataset=gs://uga-dsp/project1/train/X_train_large.txt \
    --labels=gs://uga-dsp/project1/train/y_train_large.txt \
    --testset=gs://uga-dsp/project1/test/X_test_large.txt \
    --stopwords=gs://team-crux-p1/stopwords.txt \
    -o=gs://team-crux-p1/output/