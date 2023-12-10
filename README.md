
# Инструкция по запуску проекта

## Начальная настройка

Для работы с проектом у вас должны быть установлены Docker и Docker Compose.

## Запуск проекта

Чтобы запустить проект, выполните следующие шаги:

1. Запустите Docker контейнеры:
make up

Эта команда запустит все необходимые Docker контейнеры + миграции с сидерами.

Теперь проект должен быть полностью настроен и запущен.

## Остановка проекта

Чтобы остановить проект и остановить все контейнеры Docker, выполните команду:
make down

# Запуск тестов

Для запуска тестов используйте команду:
make test

Эта команда запустит все тесты проекта в Docker контейнере.