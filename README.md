Project Description
This project presents a multimodal stress detection system that identifies whether an individual is under stress by combining physiological signals and text-based emotional cues. The system leverages deep learning and natural language processing techniques to improve reliability compared to single-modal stress detection approaches. Physiological indicators such as pulse rate, blood oxygen saturation (SpO₂), and body temperature are analyzed using a Convolutional Neural Network (CNN), while psychological stress is inferred from user-entered text using an NLP-based classifier. The final stress decision is obtained through a weighted fusion of both modalities.

Objective of the Project
The main objective of this project is to design and implement a robust and scalable stress detection framework capable of handling heterogeneous data sources. The project aims to detect physiological stress patterns using sensor data, analyze emotional stress through natural language input, and combine both predictions to produce a more accurate and reliable stress assessment. This approach reflects real-world wearable and mental-health monitoring systems, where multiple signals are used to improve decision confidence.

Physiological Data and Sensors
Physiological data is based on signals typically obtained from MAX30100 and DS18B20 sensors, including pulse rate, SpO₂, and body temperature. Due to the unavailability of physical hardware, synthetic sensor data is generated using realistic and research-backed physiological ranges for training and demonstration.

Models Used
A 1D CNN is used for physiological stress detection, achieving an accuracy of approximately 85–90%. A text-based stress detection module is implemented using TF-IDF and supervised learning, providing complementary psychological stress cues. The combined multimodal system achieves an overall accuracy of approximately 88–92%.

Multimodal Fusion Strategy
The system uses a weighted fusion strategy, assigning higher importance to physiological predictions and lower weight to text-based predictions. This approach improves robustness and compensates for limitations of individual modalities.

System Evaluation and Results
This project demonstrates an effective multimodal approach to stress detection by integrating physiological and textual information. The system is scalable and can be extended with real sensor integration, larger datasets, and advanced deep learning models.