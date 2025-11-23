### **Comprehensive Prompt: Chinese Transcript Segmentation**

**Your Role:** You are an expert data processing AI. Your primary task is to create a script that transforms a JSON object containing a Chinese transcript with word-level timestamps into a new JSON object structured by sentences.

**Primary Goal:** Read the input JSON object provided at the end of this prompt. This object contains a list of transcribed Chinese words with their individual start and end times. Your script must identify sentence boundaries based on Chinese punctuation and significant pauses, then generate a new JSON object containing a list of sentences. Each sentence object must include the complete sentence text, the start time of its first word, and the end time of its last word.

---

### **1. Input Data Structure**

The input is a single JSON object with a key (e.g., `"words"`) whose value is an array of word objects. Each word object has the following structure:

*   `word` (string): The transcribed word or punctuation mark.
*   `start` (float): The start time of the word in seconds.
*   `end` (float): The end time of the word in seconds.

---

### **2. Required Output Data Structure**

Your script must produce a single JSON object containing one key, `"sentences"`, whose value is an array of sentence objects. Each sentence object must conform to the following structure:

*   `text` (string): The full text of the sentence.
*   `start` (float): The `start` time of the very first word in the sentence.
*   `end` (float): The `end` time of the very last word or punctuation mark in the sentence.

**Example Output (based on the input below):**
```json
{
  "sentences": [
    {
      "text": "这是第一句话。",
      "start": 0.6,
      "end": 2.4
    },
    {
      "text": "你好吗？",
      "start": 3.0,
      "end": 4.1
    },
    {
      "text": "今天天气真好",
      "start": 4.5,
      "end": 6.8
    }
  ]
}
```

---

### **3. Detailed Logic and Step-by-Step Instructions**

1.  **Initialization:**
    *   Parse the input JSON data.
    *   Initialize an empty list to store the final sentence objects (e.g., `sentences_list`).
    *   Initialize an empty temporary list to accumulate word objects for the current sentence being built (e.g., `current_sentence_words`).

2.  **Iterate Through Words:**
    *   Loop through each word object in the input `words` array.
    *   Add the current word object to the `current_sentence_words` list.

3.  **Sentence Boundary Detection:**
    A sentence boundary is reached if either of the following conditions is true:
    *   **Punctuation-based:** The `word` string ends with one of the definitive Chinese sentence-ending punctuation marks:
        *   `。` (full-width period)
        *   `？` (full-width question mark)
        *   `！` (full-width exclamation mark)
    *   **Timing-based (Heuristic):** There is a significant pause between the `end` time of the current word and the `start` time of the *next* word in the array. Use a threshold of **0.75 seconds** for this pause. This is crucial for segmenting spoken Chinese which often lacks formal punctuation.

4.  **Create a Sentence Object:**
    *   When a sentence boundary is detected (from step 3), and if `current_sentence_words` is not empty, you must finalize the current sentence:
        *   **`text`**: Construct the sentence string by joining the `word` values from all objects in `current_sentence_words`. **Crucially, for Chinese, these words must be joined without any spaces.** For example, `["你", "好"]` becomes `"你好"`.
        *   **`start`**: Get the `start` time from the very first word object in `current_sentence_words`.
        *   **`end`**: Get the `end` time from the very last word object in `current_sentence_words`.
        *   Add this newly created sentence object to your `sentences_list`.
        *   Reset `current_sentence_words` to an empty list to begin the next sentence.

5.  **Handling the Final Sentence:**
    *   After the loop finishes, it's possible that `current_sentence_words` still contains words that haven't been finalized into a sentence (e.g., the transcript ends without punctuation).
    *   If `current_sentence_words` is not empty after the loop, create one last sentence object from the remaining words using the same logic as in Step 4.

6.  **Final Output:**
    *   Create the final JSON object with the key `"sentences"` and the value of your completed `sentences_list`.

---

### **Input JSON for Processing**

```json
