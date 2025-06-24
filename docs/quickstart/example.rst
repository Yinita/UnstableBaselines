# End-to-end Example

The script below trains **Qwen3-1.7B-Base** via mirror self-play on `SimpleTak` and evaluates against `google/gemini-2.0-flash-lite-001`.

.. literalinclude:: ../../example.py
\:language: python
\:linenos:
\:caption: `example.py`

Run the script with three 48 GB GPUs or scale parameters down:

.. code-block:: bash

python example.py  # \~12 h for 200 learner updates on 3x RTX6000 Ada

Launch the live dashboard in a second terminal:

.. code-block:: bash

unstable-terminal

You should see per-GPU token/s, buffer size, TrueSkill ladder, and match heat-map refresh every two seconds.