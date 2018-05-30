# Production Models
* __Xception_full__: Trained on top and finetuning ~ meh?
* __Xception_occ100000__: Trained on top and finetuning using classes that occur more than 100.000 times
* __Xception_full_latest__: Trained on top only with the Barebones generator on roughly 0.5 mil samples
    * Training
        * Loss: 0.0757
        * Accuracy: 0.9781
    * Validation (full)
        * Loss: 0.12167
        * Accuracy: 0.9702 (roughly predicts 221,2/228 classes correctly (unseen))
