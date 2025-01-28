MAKEFILE_LIST := $(MAKEFILE_LIST) Makefile

define find.functions
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
endef

help:
	@echo 'The following commands can be used.'
	@echo ''
	$(call find.functions)

start: ## Start the docker-compose services
	docker-compose up -d --build

end: ## Stop the docker-compose services
	docker-compose kill