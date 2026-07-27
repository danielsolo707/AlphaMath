# ALPHA-MATH Kaggle upload guide

If you received `AlphaMath_Kaggle_Upload_Package.zip`, extract it once on your
computer. It contains the two items Kaggle treats differently:

1. Upload `AlphaMath_Kaggle_Bundle.zip` as a Kaggle Dataset/Input.
2. In Kaggle, choose **New Notebook -> Import Notebook** and select
   `alphamath_portfolio_kaggle.ipynb`.

Then attach the uploaded code Dataset to that notebook and run its cells in order.

Also attach to the notebook:

1. Qwen2.5-Math-7B-Instruct weights as a Kaggle Model/Dataset (the automated
   CLI Kernel uses `mehedi457/qwen25-math-7b-instruct`).
2. A labeled JSON/JSONL/CSV benchmark if you want a credible accuracy report.
3. AIMO competition data if you want `submission.csv`.

The notebook auto-discovers common paths, but its configuration cell lets you
set exact paths. The last cell produces:

```text
/kaggle/working/alphamath_artifacts.zip
```

Download that archive. It contains the final Markdown report, resolved config,
package/GPU manifest, complete attempt traces, flat CSV metrics, and competition
submission when test data was available.

Important: neither ZIP contains the 7B model weights. They are too
large and must be attached separately through Kaggle.
