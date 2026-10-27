# Ключевые статьи

Отобраны по этапам из docs/HANDOFF.md. Остальные источники и подробный разбор в файлах тем.

## Мотивация и общая картина

**Müller et al. 2022**, Interspeech. Does Audio Deepfake Detection Generalize? [arXiv:2203.16263](https://arxiv.org/abs/2203.16263)
- Что: заново обучили и одинаково оценили архитектуры из литературы и собрали In-the-Wild: 58 знаменитостей и политиков, 37,9 ч живой речи и подделок из интернета. На нём ошибка моделей вырастает до 1000%.
- Нам: внешний тестовый набор и главный довод, что цифры ASVspoof не переносятся в жизнь. Вход, обрезанный до 4 с, у них хуже полной записи.

**Müller et al. 2024**, Interspeech. Harder or Different? Understanding Generalization of Audio Deepfake Detection. [arXiv:2406.03512](https://arxiv.org/abs/2406.03512)
- Что: разрыв между знакомыми и незнакомыми подделками раскладывают на «новые сложнее» и «новые другие». Почти весь разрыв из второго.
- Нам: рост модели бьёт в симптом, причина в различии генераторов. Довод за свой набор из разных свежих TTS.

**Sharma, Wang 2026**, Interspeech. VoxENES 2026: Benchmarking Generalization of Speech Spoofing Detectors Against LLM-Era TTS and Voice Conversion. [arXiv:2607.11706](https://arxiv.org/abs/2607.11706)
- Что: 10 современных TTS и VC на основе LLM плюс постобработка. Лучший из 8 готовых детекторов даёт 28,98% EER, большинство около случайного угадывания.
- Нам: ориентир для этапа 3. Настоящее падение видно на свежих TTS, а не на ASVspoof.

**Li, Ahmadiadli, Zhang 2025**, ACM Computing Surveys. A Survey on Speech Deepfake Detection. [arXiv:2404.13914](https://arxiv.org/abs/2404.13914)
- Что: больше 200 статей до марта 2024. Обобщение разложено на аугментации, функции потерь, SSL-признаки, стратегии обучения, перенос между наборами, интерпретацию и определение генератора. В открытых проблемах названа опора детекторов на тишину, низкие частоты и фон.
- Нам: главный обзор для литобзора в записке и сверки плана.

**Ojha, Li, Lee 2023**, CVPR (UnivFD). Towards Universal Fake Image Detectors that Generalize Across Generative Models. [arXiv:2302.10174](https://arxiv.org/abs/2302.10174)
- Что: изображения. Ближайший сосед на замороженном CLIP переносится на незнакомые генераторы лучше обученных детекторов, плюс 25,9% точности. Обученный детектор учит приметы подделок, а живое становится классом «всё остальное», куда и попадают незнакомые генераторы.
- Нам: формулировка причины плохого переноса и идея замороженных признаков, которую в аудио повторил Pascu 2024.

## Модель и признаки (этапы 1 и 2)

**Tak et al. 2022**, Odyssey. Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation. [arXiv:2202.12233](https://arxiv.org/abs/2202.12233), [код](https://github.com/TakHemlata/SSL_Anti-spoofing)
- Что: XLS-R 300M с головой AASIST, обучение на ASVspoof 2019 LA с аугментацией RawBoost. На ASVspoof 2021 LA 0,82% и DF 2,85% EER, лучший из трёх сидов. Выложены два чекпойнта, LA и DF.
- Нам: готовая модель этапа 1. Код берёт первые 4 с записи, поэтому начальная тишина всегда попадает в окно.

**Wang, Yamagishi 2022**, Odyssey. Investigating self-supervised front ends for speech spoofing countermeasures. [arXiv:2111.07725](https://arxiv.org/abs/2111.07725)
- Что: детектор на LFCC опирается на полосу 5,6–8 кГц, детектор на SSL на 0,1–2,4 кГц, и эта полоса полезна на всех тестовых наборах. Дообучение SSL-части даёт больше, чем заморозка.
- Нам: объяснение, почему SSL обобщается лучше. Нужно для обоснования выбора модели.

**Pascu et al. 2024**, Interspeech. Towards generalisable and calibrated audio deepfake detection with self-supervised representations. [arXiv:2309.05384](https://arxiv.org/abs/2309.05384)
- Что: замороженный XLS-R 2B и логистическая регрессия меньше чем на 2 тыс. параметров. Средний EER на восьми чужих наборах 8,8% против 30,9% у RawNet2, калибровка лучше. Чем больше SSL-модель, тем лучше.
- Нам: повторяем на этапе 2 как дешёвый детектор и сверяемся с их цифрами для XLS-R 300M: 1,0% EER на eval 2019 и 21,3% на In-the-Wild.

**El Kheir et al. 2025**, NAACL Findings. Comprehensive Layer-wise Analysis of SSL Models for Audio Deepfake Detection. [arXiv:2502.03559](https://arxiv.org/abs/2502.03559)
- Что: послойный анализ SSL-моделей на нескольких языках и типах подделок. Самые полезные нижние слои, нескольких из них хватает для конкурентного EER.
- Нам: выбор слоёв XLS-R на этапе 2 и экономия вычислений.

**Beheshti, Rostami, Chen 2026**, препринт. Probing-Guided Layer Selection from Self-Supervised Speech Models for Generalizable Audio Deepfake Detection. [arXiv:2606.30791](https://arxiv.org/abs/2606.30791)
- Что: замороженный XLS-R 300M, слои выбирают дешёвыми зондами до обучения. Четыре слоя и 1,34 млн параметров дают 4,94% EER на In-the-Wild, на 28% лучше прежнего лучшего замороженного варианта, где сливали все 25 слоёв (Xiao, Vu 2025). Зонды оценивали на тех же наборах, что служат тестом, поэтому цифра завышена.
- Нам: вне домена полезнее средние слои, а dev их перенос предсказывает плохо. Цифры для сверки только с оговоркой о выборе по тесту.

## Ярлык тишины (этапы 1 и 4)

**Müller et al. 2021**, ASVspoof 2021 Workshop. Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn? [arXiv:2106.12914](https://arxiv.org/abs/2106.12914)
- Что: в ASVspoof 2019 LA у живой речи длиннее тишина по краям. Классификатор по одной длине начальной тишины даёт 15,1% EER на eval. Обрезка тишины поднимает EER RawNet2 с 3,6% до 15,5%, а у LSTM при обрезке только в тесте с 7,4% до 35,3%.
- Нам: повторяем на этапе 1. С этой статьи начинается вся линия про тишину.

**Liu et al. 2023**, IEEE/ACM TASLP. ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild. [arXiv:2210.02437](https://arxiv.org/abs/2210.02437)
- Что: итоги ASVspoof 2021. На скрытых подмножествах без тишины EER топ-10 систем вырос с меньше 10% до больше 20%. Вырезание пауз между словами почти не меняет EER, вырезание тишины по краям поднимает его в разы: у RawNet2 9,5% против 35%. Живая речь из другого источника (VCC) даёт 15–34% EER против меньше 1%.
- Нам: ярлык сидит в тишине по краям, паузы внутри можно не трогать. Второй ярлык — источник живой речи.

**Wang, Yamagishi 2023**, ICASSP. Spoofed training data for speech spoofing countermeasure can be efficiently created using neural vocoders. [arXiv:2210.10570](https://arxiv.org/abs/2210.10570), [данные](https://zenodo.org/records/7314976)
- Что: подделки для обучения как вокодерные копии живой речи ASVspoof 2019. С ними 3,79% EER на ASVspoof 2019 с обрезанной тишиной и 7,55% на In-the-Wild. У обычного дообученного XLSR-53 обрезка тишины поднимает EER с 2,98% до 15,56%. Выложены отметки обрезки тишины LA19etrim для train, dev и eval.
- Нам: LA19etrim даёт eval без тишины на этапе 1 и обрезку обучения на этапе 4. Вокодерные копии — один из вариантов этапа 4.

**Rubio et al. 2026**, Odyssey. An Intervention-Based Framework for Shortcut Diagnosis in Spoofing Countermeasures. [arXiv:2607.03150](https://arxiv.org/abs/2607.03150)
- Что: вмешательства в неречевые участки, спектр и энергию для XLS-R с RawGAT-ST. Сильнее всего результат сдвигают вмешательства в тишину. Обрезка тишины в обучении делает модель нечувствительной к дописанной тишине, но качество на чистом ASVspoof сильно падает.
- Нам: наша модельная семья тоже опирается на тишину. Цена обрезки в обучении — главный довод проверить на этапе 4 GRL.

**Staněk et al. 2026**, Interspeech. What Do Deepfake Speech Detectors Actually Hear? [arXiv:2606.10912](https://arxiv.org/abs/2606.10912)
- Что: три детектора на WavLM с похожим EER опираются на разное, подтверждено маскированием. Голова AASIST опирается на неречевые участки и обстановку.
- Нам: у нашей головы AASIST склонность к тишине и фону, это ожидаем увидеть на этапе 1.

## Как считать ошибку (этап 1)

**Bengio, Mariéthoz 2004**, Odyssey. A Statistical Significance Test for Person Authentication. [PDF](https://publications.idiap.ch/attachments/reports/2004/bengio_2004_odyssey_stats.pdf)
- Что: HTER = (FAR + FRR) / 2 при пороге, выбранном на отдельном наборе, и формула 95% доверительного интервала для него.
- Нам: так считает наш скрипт оценки: порог в точке EER на dev, на тестах HTER с интервалом.

**Zhou, Wang 2026**, препринт. When EER Hides Deployment Failure: Auditing Threshold Transfer and Unlabeled Score Calibration for Speech Deepfake Detectors. [arXiv:2606.21584](https://arxiv.org/abs/2606.21584)
- Что: SSL-AASIST с чекпойнтом LA: 0,21% EER на части ASVspoof 2019 eval, 11,18% на In-the-Wild. С порогом из ASVspoof на In-the-Wild HTER 39,5%: отвергнуто 78,7% живых записей.
- Нам: цифры для сверки на этапе 1 и довод отчитываться не одним EER.

**Li et al. 2026**, препринт. ML-ITW: A Multilingual in-the-wild Benchmark for Speech Deepfake Detection. [arXiv:2603.05852](https://arxiv.org/abs/2603.05852)
- Что: многоязычный набор из реальной жизни. Готовая XLSR+AASIST: 0,15% EER на ASVspoof 2019 eval, 7,25% на In-the-Wild при точности 56%. Та же архитектура даёт на In-the-Wild от 2,6% до 48% только от выбора обучающих данных.
- Нам: вторая пара цифр для сверки на этапе 1. Данные влияют сильнее архитектуры.

**Dowerah et al. 2025**, препринт. Speech DF Arena: A Leaderboard for Speech DeepFake Detection Models. [arXiv:2509.02859](https://arxiv.org/abs/2509.02859)
- Что: лидерборд и код единой оценки на 14 наборах.
- Нам: формат протокола наших наборов и место, где смотреть текущий уровень моделей.

## Методы против ярлыков (этап 4)

**Xie, Cheng, Wang, Ye 2023**, Interspeech. Learning A Self-Supervised Domain-Invariant Feature Representation for Generalized Audio Deepfake Detection. [ISCA](https://www.isca-archive.org/interspeech_2023/xie23c_interspeech.html)
- Что: замороженный XLS-R, обучение на трёх наборах. GRL по корпусу только для живой речи и триплетная потеря для подделок: средний EER на 13 внешних условиях с 7,93% до 4,60%. Тишину не контролировали.
- Нам: GRL против источника живой речи уже опубликован, но без проверки, ушёл ли ярлык. Новизна этапа 4 в такой проверке, а не в самом GRL.

**Dao, Matrouf, Rouvier, Evans 2026**, препринт (IDFE). Enhancing Multi-Corpus Training in SSL-Based Anti-Spoofing Models: Domain-Invariant Feature Extraction. [arXiv:2603.18657](https://arxiv.org/abs/2603.18657)
- Что: XLS-R, GRL по корпусу при обучении на нескольких наборах, тишина обрезана и в обучении, и в тесте. Pooled EER с 6,11% до 4,88%.
- Нам: ближайший аналог этапа 4. Влияние GRL на сам ярлык они не измеряли.

**Liang et al. 2026**, SPSC. Leveraging Gradient Reversal Loss and Multitask Learning for Datasets-Aware Audio Deepfake Detection. [arXiv:2607.23961](https://arxiv.org/abs/2607.23961)
- Что: 12 обучающих наборов, оценка на 14 наборах DF Arena. GRL по метке набора почти не меняет средний EER (9,48% и 9,38%) и хуже бейзлайна на 7 наборах из 14. Та же голова без разворота градиента даёт 8,24%.
- Нам: польза GRL мала, а часть эффекта даёт сама дополнительная голова. Отсюда обязательный контроль «та же голова без разворота».

**Fursule, Kshirsagar, Avila 2026**, препринт. Towards Trustworthy Audio Deepfake Detection: A Systematic Framework for Diagnosing and Mitigating Gender Bias. [arXiv:2605.09087](https://arxiv.org/abs/2605.09087)
- Что: GRL против пола на ASVspoof 5. У модели на wav2vec 2.0 разрыв между полами сократился только потому, что EER вырос примерно на 6 п. п. у обоих полов.
- Нам: риск этапа 4: GRL может просто ухудшить модель. Проверять, ушёл ли фактор, а не только EER.

**Laakkonen, Kukanov, Hautamäki 2025**, препринт. Generalizable speech deepfake detection via meta-learned LoRA. [arXiv:2502.10838](https://arxiv.org/abs/2502.10838)
- Что: мета-обучение по атакам ASVspoof 2019, обучаются только LoRA в XLS-R с AASIST: 3,6 млн параметров вместо 318 млн. In-the-Wild с 12,15% до 6,81%, ASVspoof 5 с 23,38% до 14,10%. LoRA обгоняет полное дообучение.
- Нам: сильный и дешёвый соперник для сравнения на этапе 4 и довод за частичное дообучение.

## Свой набор свежих TTS (этап 3)

**Ge, Wang, Liu, Yamagishi 2025**, препринт (AntiDeepfake). Post-training for Deepfake Speech Detection. [arXiv:2506.21090](https://arxiv.org/abs/2506.21090), [веса](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake-nda)
- Что: дополнительное предобучение SSL на 56 тыс. ч живой и 18 тыс. ч искажённой речи. Без дообучения 1,23% EER на In-the-Wild у модели на 2 млрд параметров. Веса открыты.
- Нам: самая сильная открытая модель, прогоняем её на своём наборе. Для моделей такого уровня In-the-Wild уже не показывает разрыв.

**Klein et al. 2024**, Interspeech. Source Tracing of Audio Deepfake Systems. [arXiv:2407.08016](https://arxiv.org/abs/2407.08016)
- Что: определяют не систему синтеза целиком, а её узлы: обработку входа, акустическую модель и вокодер.
- Нам: размечаем свой набор по узлам, чтобы видеть, какие из них знакомы детектору.

**Firc et al. 2025**, Interspeech. STOPA: A Database of Systematic VariaTion Of DeePfake Audio for Open-Set Source Tracing and Attribution. [arXiv:2505.19644](https://arxiv.org/abs/2505.19644)
- Что: 700 тыс. записей от 13 синтезаторов с систематически меняющимися акустическими моделями (8) и вокодерами (6) и полными метаданными.
- Нам: образец, как планировать свой набор с контролем факторов.

**Müller, Debus 2026**, препринт. The Watermark Shortcut: How Provenance Marking Sabotages Audio Deepfake Detection. [arXiv:2606.23335](https://arxiv.org/abs/2606.23335)
- Что: если синтез помечен водяным знаком, а живая речь нет, детектор учит правило «есть знак, значит подделка». Помеченная живая речь начинает считаться подделкой, EER с 16% до 75%.
- Нам: проверяем, не ставят ли знак TTS из нашего набора, иначе измерим знак, а не синтез.
