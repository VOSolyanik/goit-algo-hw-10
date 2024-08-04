import pulp

def main():
	# Ініціалізація моделі
	model = pulp.LpProblem("Максимізація виробництва", pulp.LpMaximize)

	# Визначення змінних
	A = pulp.LpVariable('Лимонад', lowBound=0, cat='Integer')  # Кількість продукту А
	B = pulp.LpVariable('Фруктовий сік', lowBound=0,  cat='Integer')  # Кількість продукту Б

	# Функція цілі (Максимізація прибутку)
	model += A + B, "Максимізація кількості продуктів"

	# Додавання обмежень
	model += 2 * A + 1 * B <= 100  # Обмеження Води
	model += 1 * A <= 50  		   # Обмеження Цукру
	model += 1 * A <= 30  		   # Обмеження Лимонного соку
	model += 1 * B <= 40  		   # Обмеження Фруктового пюре

	# Розв'язання моделі
	model.solve()

	# Вивід результатів
	print("Виробляти Лимонад:", A.varValue)
	print("Виробляти Фруктовий сік:", B.varValue)

if __name__ == "__main__":
    main()