/**
 * Cortex shadcn/ui component library
 * All components use CSS variables from index.css for consistent theming.
 */

import React, { forwardRef, createContext, useContext, useState } from "react";
import * as RadixDialog from "@radix-ui/react-dialog";
import * as RadixTabs from "@radix-ui/react-tabs";
import * as RadixSelect from "@radix-ui/react-select";
import * as RadixSwitch from "@radix-ui/react-switch";
import * as RadixProgress from "@radix-ui/react-progress";
import * as RadixSeparator from "@radix-ui/react-separator";
import * as RadixTooltip from "@radix-ui/react-tooltip";
import * as RadixDropdownMenu from "@radix-ui/react-dropdown-menu";
import * as RadixScrollArea from "@radix-ui/react-scroll-area";
import * as RadixAvatar from "@radix-ui/react-avatar";
import * as RadixCollapsible from "@radix-ui/react-collapsible";
import { Check, ChevronDown, X } from "lucide-react";

// ── cn utility ───────────────────────────────────────────────────────────────
export function cn(...classes) {
  return classes.filter(Boolean).join(" ");
}

// ─────────────────────────────────────────────────────────────────────────────
// BUTTON
// ─────────────────────────────────────────────────────────────────────────────

const buttonVariants = {
  default:     "bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm",
  destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
  outline:     "border border-border bg-transparent hover:bg-accent hover:text-accent-foreground",
  secondary:   "bg-secondary text-secondary-foreground hover:bg-secondary/80",
  ghost:       "hover:bg-accent hover:text-accent-foreground",
  link:        "text-primary underline-offset-4 hover:underline",
};

const buttonSizes = {
  default: "h-9 px-4 py-2 text-sm",
  sm:      "h-8 px-3 text-xs",
  lg:      "h-10 px-6 text-sm",
  icon:    "h-9 w-9",
};

export const Button = forwardRef(({ className, variant = "default", size = "default", ...props }, ref) => (
  <button
    ref={ref}
    className={cn(
      "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md font-medium",
      "transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
      "disabled:pointer-events-none disabled:opacity-50",
      buttonVariants[variant],
      buttonSizes[size],
      className
    )}
    {...props}
  />
));
Button.displayName = "Button";

// ─────────────────────────────────────────────────────────────────────────────
// CARD
// ─────────────────────────────────────────────────────────────────────────────

export const Card = forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("rounded-lg border border-border bg-card text-card-foreground shadow-sm", className)} {...props} />
));
Card.displayName = "Card";

export const CardHeader = forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex flex-col space-y-1 p-6", className)} {...props} />
));
CardHeader.displayName = "CardHeader";

export const CardTitle = forwardRef(({ className, ...props }, ref) => (
  <h3 ref={ref} className={cn("text-sm font-semibold leading-none tracking-tight", className)} {...props} />
));
CardTitle.displayName = "CardTitle";

export const CardDescription = forwardRef(({ className, ...props }, ref) => (
  <p ref={ref} className={cn("text-xs text-muted-foreground", className)} {...props} />
));
CardDescription.displayName = "CardDescription";

export const CardContent = forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
));
CardContent.displayName = "CardContent";

export const CardFooter = forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex items-center p-6 pt-0", className)} {...props} />
));
CardFooter.displayName = "CardFooter";

// ─────────────────────────────────────────────────────────────────────────────
// INPUT
// ─────────────────────────────────────────────────────────────────────────────

export const Input = forwardRef(({ className, type, ...props }, ref) => (
  <input
    type={type}
    ref={ref}
    className={cn(
      "flex h-9 w-full rounded-md border border-border",
      "bg-input px-3 py-1 text-sm text-foreground",
      "placeholder:text-muted-foreground",
      "focus:outline-none focus:ring-1 focus:ring-ring",
      "disabled:cursor-not-allowed disabled:opacity-50",
      "transition-colors",
      className
    )}
    {...props}
  />
));
Input.displayName = "Input";

// ─────────────────────────────────────────────────────────────────────────────
// TEXTAREA
// ─────────────────────────────────────────────────────────────────────────────

export const Textarea = forwardRef(({ className, ...props }, ref) => (
  <textarea
    ref={ref}
    className={cn(
      "flex w-full rounded-md border border-border",
      "bg-input px-3 py-2 text-sm text-foreground",
      "placeholder:text-muted-foreground",
      "focus:outline-none focus:ring-1 focus:ring-ring",
      "disabled:cursor-not-allowed disabled:opacity-50",
      "transition-colors resize-none",
      className
    )}
    {...props}
  />
));
Textarea.displayName = "Textarea";

// ─────────────────────────────────────────────────────────────────────────────
// LABEL
// ─────────────────────────────────────────────────────────────────────────────

export const Label = forwardRef(({ className, ...props }, ref) => (
  <label
    ref={ref}
    className={cn("text-xs font-medium text-foreground leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70", className)}
    {...props}
  />
));
Label.displayName = "Label";

// ─────────────────────────────────────────────────────────────────────────────
// BADGE
// ─────────────────────────────────────────────────────────────────────────────

const badgeVariants = {
  default:     "bg-primary/15 text-primary border-primary/25",
  secondary:   "bg-secondary text-secondary-foreground border-border",
  destructive: "bg-destructive/15 text-destructive border-destructive/25",
  outline:     "border border-border text-foreground",
  success:     "bg-emerald-500/15 text-emerald-400 border-emerald-500/25",
  warning:     "bg-amber-500/15 text-amber-400 border-amber-500/25",
  cyan:        "bg-cyan-500/15 text-cyan-400 border-cyan-500/25",
};

export function Badge({ className, variant = "default", ...props }) {
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium transition-colors", badgeVariants[variant], className)} {...props} />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SEPARATOR
// ─────────────────────────────────────────────────────────────────────────────

export const Separator = forwardRef(({ className, orientation = "horizontal", ...props }, ref) => (
  <RadixSeparator.Root
    ref={ref}
    orientation={orientation}
    className={cn("shrink-0 bg-border", orientation === "horizontal" ? "h-px w-full" : "h-full w-px", className)}
    {...props}
  />
));
Separator.displayName = "Separator";

// ─────────────────────────────────────────────────────────────────────────────
// PROGRESS
// ─────────────────────────────────────────────────────────────────────────────

export const Progress = forwardRef(({ className, value, ...props }, ref) => (
  <RadixProgress.Root
    ref={ref}
    className={cn("relative h-1.5 w-full overflow-hidden rounded-full bg-muted", className)}
    {...props}
  >
    <RadixProgress.Indicator
      className="h-full bg-primary transition-all duration-300 ease-in-out"
      style={{ transform: `translateX(-${100 - (value || 0)}%)` }}
    />
  </RadixProgress.Root>
));
Progress.displayName = "Progress";

// ─────────────────────────────────────────────────────────────────────────────
// SWITCH
// ─────────────────────────────────────────────────────────────────────────────

export const Switch = forwardRef(({ className, ...props }, ref) => (
  <RadixSwitch.Root
    ref={ref}
    className={cn(
      "peer inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent",
      "transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
      "disabled:cursor-not-allowed disabled:opacity-50",
      "data-[state=checked]:bg-primary data-[state=unchecked]:bg-muted",
      className
    )}
    {...props}
  >
    <RadixSwitch.Thumb className={cn(
      "pointer-events-none block h-4 w-4 rounded-full bg-white shadow-sm transition-transform",
      "data-[state=checked]:translate-x-4 data-[state=unchecked]:translate-x-0"
    )} />
  </RadixSwitch.Root>
));
Switch.displayName = "Switch";

// ─────────────────────────────────────────────────────────────────────────────
// TABS
// ─────────────────────────────────────────────────────────────────────────────

export const Tabs = RadixTabs.Root;

export const TabsList = forwardRef(({ className, ...props }, ref) => (
  <RadixTabs.List
    ref={ref}
    className={cn("inline-flex h-9 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground", className)}
    {...props}
  />
));
TabsList.displayName = "TabsList";

export const TabsTrigger = forwardRef(({ className, ...props }, ref) => (
  <RadixTabs.Trigger
    ref={ref}
    className={cn(
      "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1 text-xs font-medium transition-all",
      "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
      "disabled:pointer-events-none disabled:opacity-50",
      "data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm",
      className
    )}
    {...props}
  />
));
TabsTrigger.displayName = "TabsTrigger";

// Override Radix's default `display: block` so that flex classes passed
// via className actually work (e.g. flex-1 flex flex-col on the content pane).
export const TabsContent = forwardRef(({ className, ...props }, ref) => (
  <RadixTabs.Content
    ref={ref}
    style={{ display: undefined }}   // let Radix handle visibility via hidden attr
    className={cn("mt-2 focus-visible:outline-none [&:not([hidden])]:flex [&:not([hidden])]:flex-col", className)}
    {...props}
  />
));
TabsContent.displayName = "TabsContent";

// ─────────────────────────────────────────────────────────────────────────────
// SELECT
// ─────────────────────────────────────────────────────────────────────────────

export const Select = RadixSelect.Root;
export const SelectGroup = RadixSelect.Group;
export const SelectValue = RadixSelect.Value;

export const SelectTrigger = forwardRef(({ className, children, ...props }, ref) => (
  <RadixSelect.Trigger
    ref={ref}
    className={cn(
      "flex h-9 w-full items-center justify-between rounded-md border border-border",
      "bg-input px-3 py-2 text-sm text-foreground",
      "placeholder:text-muted-foreground",
      "focus:outline-none focus:ring-1 focus:ring-ring",
      "disabled:cursor-not-allowed disabled:opacity-50",
      "[&>span]:line-clamp-1",
      className
    )}
    {...props}
  >
    {children}
    <RadixSelect.Icon asChild>
      <ChevronDown className="h-3.5 w-3.5 opacity-50" />
    </RadixSelect.Icon>
  </RadixSelect.Trigger>
));
SelectTrigger.displayName = "SelectTrigger";

export const SelectContent = forwardRef(({ className, children, position = "popper", ...props }, ref) => (
  <RadixSelect.Portal>
    <RadixSelect.Content
      ref={ref}
      className={cn(
        "relative z-50 min-w-[8rem] overflow-hidden rounded-md border border-border",
        "bg-popover text-popover-foreground shadow-lg",
        "data-[state=open]:animate-fade-in-scale",
        position === "popper" && "translate-y-1",
        className
      )}
      position={position}
      {...props}
    >
      <RadixSelect.Viewport className={cn("p-1", position === "popper" && "w-full min-w-[var(--radix-select-trigger-width)]")}>
        {children}
      </RadixSelect.Viewport>
    </RadixSelect.Content>
  </RadixSelect.Portal>
));
SelectContent.displayName = "SelectContent";

export const SelectItem = forwardRef(({ className, children, ...props }, ref) => (
  <RadixSelect.Item
    ref={ref}
    className={cn(
      "relative flex w-full cursor-pointer select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm",
      "outline-none focus:bg-accent focus:text-accent-foreground",
      "data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className
    )}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <RadixSelect.ItemIndicator>
        <Check className="h-3 w-3" />
      </RadixSelect.ItemIndicator>
    </span>
    <RadixSelect.ItemText>{children}</RadixSelect.ItemText>
  </RadixSelect.Item>
));
SelectItem.displayName = "SelectItem";

// ─────────────────────────────────────────────────────────────────────────────
// DIALOG / MODAL
// ─────────────────────────────────────────────────────────────────────────────

export const Dialog = RadixDialog.Root;
export const DialogTrigger = RadixDialog.Trigger;

export const DialogPortal = RadixDialog.Portal;

export const DialogOverlay = forwardRef(({ className, ...props }, ref) => (
  <RadixDialog.Overlay
    ref={ref}
    className={cn("fixed inset-0 z-50 bg-black/60 backdrop-blur-sm data-[state=open]:animate-fade-in", className)}
    {...props}
  />
));
DialogOverlay.displayName = "DialogOverlay";

export const DialogContent = forwardRef(({ className, children, ...props }, ref) => (
  <RadixDialog.Portal>
    <DialogOverlay />
    <RadixDialog.Content
      ref={ref}
      className={cn(
        "fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2",
        "w-full max-w-lg rounded-lg border border-border bg-card p-6 shadow-xl",
        "data-[state=open]:animate-fade-in-scale",
        className
      )}
      {...props}
    >
      {children}
      <RadixDialog.Close className="absolute right-4 top-4 rounded-sm opacity-70 transition-opacity hover:opacity-100 focus:outline-none">
        <X className="h-4 w-4" />
      </RadixDialog.Close>
    </RadixDialog.Content>
  </RadixDialog.Portal>
));
DialogContent.displayName = "DialogContent";

export const DialogHeader = ({ className, ...props }) => (
  <div className={cn("flex flex-col space-y-1.5 mb-4", className)} {...props} />
);

export const DialogTitle = forwardRef(({ className, ...props }, ref) => (
  <RadixDialog.Title ref={ref} className={cn("text-base font-semibold", className)} {...props} />
));
DialogTitle.displayName = "DialogTitle";

export const DialogDescription = forwardRef(({ className, ...props }, ref) => (
  <RadixDialog.Description ref={ref} className={cn("text-sm text-muted-foreground", className)} {...props} />
));
DialogDescription.displayName = "DialogDescription";

// ─────────────────────────────────────────────────────────────────────────────
// DROPDOWN MENU
// ─────────────────────────────────────────────────────────────────────────────

export const DropdownMenu       = RadixDropdownMenu.Root;
export const DropdownMenuTrigger= RadixDropdownMenu.Trigger;

export const DropdownMenuContent = forwardRef(({ className, sideOffset = 4, ...props }, ref) => (
  <RadixDropdownMenu.Portal>
    <RadixDropdownMenu.Content
      ref={ref}
      sideOffset={sideOffset}
      className={cn(
        "z-50 min-w-[180px] overflow-hidden rounded-md border border-border bg-popover p-1 shadow-md",
        "data-[state=open]:animate-fade-in-scale",
        className
      )}
      {...props}
    />
  </RadixDropdownMenu.Portal>
));
DropdownMenuContent.displayName = "DropdownMenuContent";

export const DropdownMenuItem = forwardRef(({ className, inset, ...props }, ref) => (
  <RadixDropdownMenu.Item
    ref={ref}
    className={cn(
      "relative flex cursor-pointer select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none",
      "transition-colors focus:bg-accent focus:text-accent-foreground",
      "data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      inset && "pl-8",
      className
    )}
    {...props}
  />
));
DropdownMenuItem.displayName = "DropdownMenuItem";

export const DropdownMenuSeparator = forwardRef(({ className, ...props }, ref) => (
  <RadixDropdownMenu.Separator ref={ref} className={cn("-mx-1 my-1 h-px bg-border", className)} {...props} />
));
DropdownMenuSeparator.displayName = "DropdownMenuSeparator";

export const DropdownMenuLabel = forwardRef(({ className, ...props }, ref) => (
  <RadixDropdownMenu.Label ref={ref} className={cn("px-2 py-1.5 text-xs font-medium text-muted-foreground", className)} {...props} />
));
DropdownMenuLabel.displayName = "DropdownMenuLabel";

// ─────────────────────────────────────────────────────────────────────────────
// TOOLTIP
// ─────────────────────────────────────────────────────────────────────────────

export const TooltipProvider = RadixTooltip.Provider;
export const Tooltip          = RadixTooltip.Root;
export const TooltipTrigger   = RadixTooltip.Trigger;

export const TooltipContent = forwardRef(({ className, sideOffset = 4, ...props }, ref) => (
  <RadixTooltip.Content
    ref={ref}
    sideOffset={sideOffset}
    className={cn("z-50 overflow-hidden rounded-md border border-border bg-popover px-2.5 py-1.5 text-xs text-foreground shadow-md animate-fade-in", className)}
    {...props}
  />
));
TooltipContent.displayName = "TooltipContent";

// ─────────────────────────────────────────────────────────────────────────────
// SCROLL AREA
// ─────────────────────────────────────────────────────────────────────────────

export const ScrollArea = forwardRef(({ className, children, ...props }, ref) => (
  <RadixScrollArea.Root ref={ref} className={cn("relative overflow-hidden", className)} {...props}>
    <RadixScrollArea.Viewport className="h-full w-full rounded-[inherit]">
      {children}
    </RadixScrollArea.Viewport>
    <RadixScrollArea.Scrollbar orientation="vertical" className="flex w-1.5 touch-none select-none p-px transition-colors">
      <RadixScrollArea.Thumb className="relative flex-1 rounded-full bg-border" />
    </RadixScrollArea.Scrollbar>
    <RadixScrollArea.Corner />
  </RadixScrollArea.Root>
));
ScrollArea.displayName = "ScrollArea";

// ─────────────────────────────────────────────────────────────────────────────
// AVATAR
// ─────────────────────────────────────────────────────────────────────────────

export const Avatar = forwardRef(({ className, ...props }, ref) => (
  <RadixAvatar.Root ref={ref} className={cn("relative flex h-8 w-8 shrink-0 overflow-hidden rounded-full", className)} {...props} />
));
Avatar.displayName = "Avatar";

export const AvatarImage = forwardRef(({ className, ...props }, ref) => (
  <RadixAvatar.Image ref={ref} className={cn("aspect-square h-full w-full", className)} {...props} />
));
AvatarImage.displayName = "AvatarImage";

export const AvatarFallback = forwardRef(({ className, ...props }, ref) => (
  <RadixAvatar.Fallback ref={ref} className={cn("flex h-full w-full items-center justify-center rounded-full bg-muted text-xs font-medium", className)} {...props} />
));
AvatarFallback.displayName = "AvatarFallback";

// ─────────────────────────────────────────────────────────────────────────────
// COLLAPSIBLE
// ─────────────────────────────────────────────────────────────────────────────

export const Collapsible       = RadixCollapsible.Root;
export const CollapsibleTrigger= RadixCollapsible.Trigger;
export const CollapsibleContent= RadixCollapsible.Content;

// ─────────────────────────────────────────────────────────────────────────────
// SKELETON
// ─────────────────────────────────────────────────────────────────────────────

export function Skeleton({ className, ...props }) {
  return <div className={cn("animate-pulse rounded-md bg-muted", className)} {...props} />;
}

// ─────────────────────────────────────────────────────────────────────────────
// STAT CARD  (custom compound component)
// ─────────────────────────────────────────────────────────────────────────────

export function StatCard({ icon: Icon, label, value, delta, description, color = "text-primary", className }) {
  const isPositive = delta > 0;
  return (
    <Card className={cn("p-5", className)}>
      <div className="flex items-start justify-between mb-3">
        <div className={cn("p-1.5 rounded-md", color === "text-primary" ? "bg-primary/10" : "bg-muted")}>
          <Icon className={cn("h-4 w-4", color)} />
        </div>
        {delta !== undefined && (
          <span className={cn("text-[11px] font-medium", isPositive ? "text-emerald-400" : "text-red-400")}>
            {isPositive ? "+" : ""}{(delta * 100).toFixed(0)}%
          </span>
        )}
      </div>
      <div className={cn("text-2xl font-bold tracking-tight mb-0.5", color)}>{value}</div>
      <p className="text-xs font-medium text-foreground">{label}</p>
      {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
    </Card>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// EMPTY STATE
// ─────────────────────────────────────────────────────────────────────────────

export function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="p-3 rounded-full bg-muted mb-4">
        <Icon className="h-6 w-6 text-muted-foreground" />
      </div>
      <h3 className="text-sm font-semibold mb-1">{title}</h3>
      {description && <p className="text-xs text-muted-foreground mb-4 max-w-xs">{description}</p>}
      {action}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGE HEADER
// ─────────────────────────────────────────────────────────────────────────────

export function PageHeader({ title, description, actions, className }) {
  return (
    <div className={cn("flex items-start justify-between pb-5 border-b border-border", className)}>
      <div>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="text-sm text-muted-foreground mt-1">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2 ml-4">{actions}</div>}
    </div>
  );
}
