MODULES := logit linear array
PGPORTDIR := src/ports/postgres/
MODULEDIRS := $(addprefix $(PGPORTDIR),$(MODULES))
PGMODULES := $(MODULEDIRS:%=%-pg)

.PHONY: pg $(PGMODULES)
	
pg: $(PGMODULES) 

$(PGMODULES):
	make -C $(@:%-pg=%) pg

SQLFILES := $(foreach dir,$(MODULEDIRS),$(shell find $(dir) -name 'create*.sql')) 
MODELTABLES := $(foreach dir,$(MODULEDIRS),$(shell find $(dir) -name '*model.sql')) 
AGGREGATES := $(foreach dir,$(MODULEDIRS),$(shell find $(dir) -name 'rmse.sql')) 
ARRAY_FUNCS := $(foreach dir,$(MODULEDIRS),$(shell find $(dir) -name 'array.sql')) 

.PHONY: install-pg $(SQLFILES) $(MODELTABLES) $(ARRAY_FUNCS)

install-pg: $(SQLFILES)

$(SQLFILES): $(ARRAY_FUNCS) $(MODELTABLES) $(AGGREGATES)
	psql -f $@

$(MODELTABLES):
	psql -f $@

$(ARRAY_FUNCS):
	psql -f $@

