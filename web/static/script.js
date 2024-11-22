const form = document.getElementById('survey-form');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    try {
        const response = await fetch('/submit', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        resultDiv.innerHTML = `<h2>Результат:</h2><p>${data.result}</p>`; // Заполняем div результатом

    } catch (error) {
        console.error('Ошибка:', error);
        resultDiv.innerHTML = `<p>Произошла ошибка. Пожалуйста, попробуйте еще раз.</p>`;
    }
});


fetch('../static/questions.json') // Замените 'questions.json' на путь к вашему файлу
  .then(response => response.json())
  .then(questions => {
    const quizContainer = document.getElementById('survey-form'); // ID контейнера для опроса в HTML
    const referenceElement = quizContainer.children[0];
    var question_count = 1;
    questions.forEach(question => {
      const questionElement = document.createElement('div');
      questionElement.className = 'question';
      questionElement.innerHTML = `<p>${question.question}</p>`;

      var answersElement;
      if (question.type == 'select') {
        answersElement = document.createElement('select');
        answersElement.name = 'q' + String(question_count);
      }
      else {
        answersElement = document.createElement('div');
      }

      answersElement.className = 'options';
      question.options.forEach(answer => {
        var input;
        if (question.type == 'select') {
          input = document.createElement('option');
          input.text = answer;
          input.value = answer;
          // input.name = 'q' + String(question_count);

          answersElement.appendChild(input);
        }
        else {
          input = document.createElement('input');
          input.type = question.type;
          const label = document.createElement('label');
  
          input.name = 'q' + String(question_count); // Важно для группировки ответов
          if (question.type == 'radio') {
            label.textContent = answer;
            input.value = answer;
          }
          else {
            input.value = "";
            input.placeholder = "Пиши";
          }
          input.id = input.name + "_" + input.value; // Уникальный ID для каждого ответа
          label.htmlFor = input.id;
          
  
          answersElement.appendChild(input);
          answersElement.appendChild(label);
        }
      });

      question_count += 1;

      questionElement.appendChild(answersElement);
      quizContainer.insertBefore(questionElement, referenceElement);
    });
  })
  .catch(error => console.error('Ошибка загрузки JSON:', error));